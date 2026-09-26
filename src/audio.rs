use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, SupportedStreamConfig};
use rtrb::{Consumer, Producer};

pub const TARGET_SAMPLE_RATE: u32 = 16000;

/// Thread-safe metrics and controls shared between audio threads, worker thread, and UI
#[derive(Debug)]
pub struct AudioControls {
    pub enabled: AtomicBool,
    pub input_gain_bits: AtomicU32,
    pub output_volume_bits: AtomicU32,
    pub input_noise_gate_bits: AtomicU32,
    pub output_noise_gate_bits: AtomicU32,
    pub input_rms_bits: AtomicU32,
    pub output_rms_bits: AtomicU32,
    pub rtf_bits: AtomicU32,
    pub latency_ms_bits: AtomicU32,
}

impl Default for AudioControls {
    fn default() -> Self {
        Self {
            enabled: AtomicBool::new(true),
            input_gain_bits: AtomicU32::new(1.0f32.to_bits()),
            output_volume_bits: AtomicU32::new(1.0f32.to_bits()),
            input_noise_gate_bits: AtomicU32::new(0.01f32.to_bits()),
            output_noise_gate_bits: AtomicU32::new(0.0f32.to_bits()),
            input_rms_bits: AtomicU32::new(0.0f32.to_bits()),
            output_rms_bits: AtomicU32::new(0.0f32.to_bits()),
            rtf_bits: AtomicU32::new(0.0f32.to_bits()),
            latency_ms_bits: AtomicU32::new(0.0f32.to_bits()),
        }
    }
}

impl AudioControls {
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn set_enabled(&self, val: bool) {
        self.enabled.store(val, Ordering::Relaxed);
    }

    pub fn input_gain(&self) -> f32 {
        f32::from_bits(self.input_gain_bits.load(Ordering::Relaxed))
    }

    pub fn set_input_gain(&self, val: f32) {
        self.input_gain_bits.store(val.to_bits(), Ordering::Relaxed);
    }

    pub fn output_volume(&self) -> f32 {
        f32::from_bits(self.output_volume_bits.load(Ordering::Relaxed))
    }

    pub fn set_output_volume(&self, val: f32) {
        self.output_volume_bits.store(val.to_bits(), Ordering::Relaxed);
    }

    pub fn input_noise_gate(&self) -> f32 {
        f32::from_bits(self.input_noise_gate_bits.load(Ordering::Relaxed))
    }

    pub fn set_input_noise_gate(&self, val: f32) {
        self.input_noise_gate_bits.store(val.to_bits(), Ordering::Relaxed);
    }

    pub fn output_noise_gate(&self) -> f32 {
        f32::from_bits(self.output_noise_gate_bits.load(Ordering::Relaxed))
    }

    pub fn set_output_noise_gate(&self, val: f32) {
        self.output_noise_gate_bits.store(val.to_bits(), Ordering::Relaxed);
    }

    pub fn input_rms(&self) -> f32 {
        f32::from_bits(self.input_rms_bits.load(Ordering::Relaxed))
    }

    pub fn set_input_rms(&self, val: f32) {
        self.input_rms_bits.store(val.to_bits(), Ordering::Relaxed);
    }

    pub fn output_rms(&self) -> f32 {
        f32::from_bits(self.output_rms_bits.load(Ordering::Relaxed))
    }

    pub fn set_output_rms(&self, val: f32) {
        self.output_rms_bits.store(val.to_bits(), Ordering::Relaxed);
    }

    pub fn rtf(&self) -> f32 {
        f32::from_bits(self.rtf_bits.load(Ordering::Relaxed))
    }

    pub fn set_rtf(&self, val: f32) {
        self.rtf_bits.store(val.to_bits(), Ordering::Relaxed);
    }

    pub fn latency_ms(&self) -> f32 {
        f32::from_bits(self.latency_ms_bits.load(Ordering::Relaxed))
    }

    pub fn set_latency_ms(&self, val: f32) {
        self.latency_ms_bits.store(val.to_bits(), Ordering::Relaxed);
    }
}

pub struct AudioSystem {
    pub input_stream: Stream,
    pub output_stream: Stream,
    pub input_device_name: String,
    pub output_device_name: String,
}

impl AudioSystem {
    pub fn init(
        input_producer: Producer<f32>,
        mut output_consumer: Consumer<f32>,
        controls: Arc<AudioControls>,
        custom_input_device: Option<&str>,
        custom_output_device: Option<&str>,
    ) -> Result<Self> {
        let host = cpal::default_host();

        // 1. Select input device (default mic or matching name)
        let input_device = if let Some(dev_name) = custom_input_device {
            host.input_devices()?
                .find(|d| d.name().map(|n| n.contains(dev_name)).unwrap_or(false))
                .context(format!("Could not find input microphone matching '{}'", dev_name))?
        } else {
            host.default_input_device()
                .context("No default audio input device (microphone) found")?
        };

        let in_dev_name = input_device.name().unwrap_or_else(|_| "Default Microphone".into());
        println!("Selected Input Device (Microphone): {}", in_dev_name);

        // 2. Select output device (default speaker/headphones or matching name)
        let output_device = if let Some(dev_name) = custom_output_device {
            host.output_devices()?
                .find(|d| d.name().map(|n| n.contains(dev_name)).unwrap_or(false))
                .context(format!("Could not find output device matching '{}'", dev_name))?
        } else {
            host.default_output_device()
                .context("No default audio output device (speaker) found")?
        };

        let out_dev_name = output_device.name().unwrap_or_else(|_| "Default Speaker".into());
        println!("Selected Output Device (Speaker): {}", out_dev_name);

        // 3. Configure Input Stream
        let input_config = Self::get_stream_config(&input_device, true)?;
        let in_channels = input_config.channels();
        let in_rate = input_config.sample_rate().0;
        let mut in_producer = input_producer;
        let in_controls = Arc::clone(&controls);

        let mut in_resample_pos = 0.0f64;
        let in_step = in_rate as f64 / TARGET_SAMPLE_RATE as f64;
        let mut prev_mono_sample = 0.0f32;

        let input_stream = input_device.build_input_stream(
            &input_config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let gain = in_controls.input_gain();
                let mut sum_sq = 0.0f32;
                let mut sample_count = 0;

                let num_frames = data.len() / in_channels as usize;
                for frame_idx in 0..num_frames {
                    let mut frame_mono = 0.0f32;
                    for ch in 0..in_channels as usize {
                        frame_mono += data[frame_idx * in_channels as usize + ch];
                    }
                    frame_mono = (frame_mono / in_channels as f32) * gain;
                    sum_sq += frame_mono * frame_mono;
                    sample_count += 1;

                    // Dynamic Noise Gate: Mute audio if the previous chunk RMS was below the threshold
                    // This prevents the AI model from amplifying static artifacts.
                    let current_rms = in_controls.input_rms();
                    let in_gate = in_controls.input_noise_gate();
                    if current_rms < in_gate && in_gate > 0.0001 {
                        // Apply soft decay to zero to prevent popping
                        frame_mono *= (current_rms / in_gate).powi(2);
                    }

                    if in_rate == TARGET_SAMPLE_RATE {
                        let _ = in_producer.push(frame_mono);
                    } else {
                        while in_resample_pos < 1.0 {
                            let interpolated = prev_mono_sample + (frame_mono - prev_mono_sample) * (in_resample_pos as f32);
                            let _ = in_producer.push(interpolated);
                            in_resample_pos += in_step;
                        }
                        in_resample_pos -= 1.0;
                        prev_mono_sample = frame_mono;
                    }
                }

                if sample_count > 0 {
                    let rms = (sum_sq / sample_count as f32).sqrt();
                    in_controls.set_input_rms(rms);
                }
            },
            |err| eprintln!("Input audio stream error: {}", err),
            None,
        ).context("Failed to build input audio stream")?;

        // 4. Configure Output Stream (directly to default speaker/headphones)
        let output_config = Self::get_stream_config(&output_device, false)?;
        let out_channels = output_config.channels();
        let out_rate = output_config.sample_rate().0;
        let out_controls = Arc::clone(&controls);

        let mut out_resample_pos = 0.0f64;
        let out_step = TARGET_SAMPLE_RATE as f64 / out_rate as f64;
        let mut curr_out_sample = 0.0f32;
        let mut next_out_sample = 0.0f32;

        let output_stream = output_device.build_output_stream(
            &output_config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let volume = out_controls.output_volume();
                let mut sum_sq = 0.0f32;
                let mut sample_count = 0;

                let num_frames = data.len() / out_channels as usize;
                for frame_idx in 0..num_frames {
                    let mut target_sample = if out_rate == TARGET_SAMPLE_RATE {
                        output_consumer.pop().unwrap_or(0.0) * volume
                    } else {
                        while out_resample_pos >= 1.0 {
                            curr_out_sample = next_out_sample;
                            next_out_sample = output_consumer.pop().unwrap_or(0.0) * volume;
                            out_resample_pos -= 1.0;
                        }
                        let s = curr_out_sample + (next_out_sample - curr_out_sample) * (out_resample_pos as f32);
                        out_resample_pos += out_step;
                        s
                    };

                    let out_gate = out_controls.output_noise_gate();
                    let current_out_rms = out_controls.output_rms();
                    if current_out_rms < out_gate && out_gate > 0.0001 {
                        target_sample *= (current_out_rms / out_gate).powi(2);
                    }

                    // Soft clipper to prevent distortion and allow high gain
                    target_sample = target_sample / (1.0 + target_sample.abs());

                    sum_sq += target_sample * target_sample;
                    sample_count += 1;

                    for ch in 0..out_channels as usize {
                        data[frame_idx * out_channels as usize + ch] = target_sample;
                    }
                }

                if sample_count > 0 {
                    let rms = (sum_sq / sample_count as f32).sqrt();
                    out_controls.set_output_rms(rms);
                }
            },
            |err| eprintln!("Output audio stream error: {}", err),
            None,
        ).context("Failed to build output audio stream")?;

        input_stream.play().context("Failed to start input stream")?;
        output_stream.play().context("Failed to start output stream")?;

        println!("Direct audio pipeline running: Mic -> Voice Changer -> Speaker");

        Ok(Self {
            input_stream,
            output_stream,
            input_device_name: in_dev_name,
            output_device_name: out_dev_name,
        })
    }

    fn get_stream_config(device: &Device, is_input: bool) -> Result<SupportedStreamConfig> {
        if is_input {
            if let Ok(configs) = device.supported_input_configs() {
                for config in configs {
                    if config.sample_format() == SampleFormat::F32
                        && config.min_sample_rate().0 <= TARGET_SAMPLE_RATE
                        && config.max_sample_rate().0 >= TARGET_SAMPLE_RATE
                    {
                        return Ok(config.with_sample_rate(cpal::SampleRate(TARGET_SAMPLE_RATE)));
                    }
                }
            }
            Ok(device.default_input_config()?)
        } else {
            if let Ok(configs) = device.supported_output_configs() {
                for config in configs {
                    if config.sample_format() == SampleFormat::F32
                        && config.min_sample_rate().0 <= TARGET_SAMPLE_RATE
                        && config.max_sample_rate().0 >= TARGET_SAMPLE_RATE
                    {
                        return Ok(config.with_sample_rate(cpal::SampleRate(TARGET_SAMPLE_RATE)));
                    }
                }
            }
            Ok(device.default_output_config()?)
        }
    }
}
