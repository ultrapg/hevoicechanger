use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use anyhow::Result;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait};
use rtrb::RingBuffer;

use hevoicechanger::audio::{AudioControls, AudioSystem};
use hevoicechanger::engine::{VoiceChangerEngine, DEFAULT_CHUNK_SIZE, DEFAULT_HISTORY_SIZE, WorkerCommand, ExecutionBackend};
use hevoicechanger::ui::VoiceChangerApp;

#[derive(Parser, Debug)]
#[command(author, version, about = "Ultra-low latency Linux AI Voice Changer (Direct Mic to Speaker)")]
struct Args {
    /// Path to initial ONNX voice model
    #[arg(short, long, default_value = "models/female_voice_hfg_new.onnx")]
    model: PathBuf,

    /// Specific microphone device name to use (defaults to system default)
    #[arg(short, long)]
    input_device: Option<String>,

    /// Specific speaker/output device name to use (defaults to system default)
    #[arg(short, long)]
    output_device: Option<String>,

    /// Initial mic input gain multiplier
    #[arg(short, long, default_value_t = 1.0)]
    gain: f32,

    /// Initial voice output volume multiplier
    #[arg(short, long, default_value_t = 1.0)]
    volume: f32,

    /// Run in headless daemon mode (no GUI window)
    #[arg(long)]
    headless: bool,

    /// List available audio devices and exit
    #[arg(long)]
    list_devices: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 1. If user requested device listing
    if args.list_devices {
        list_audio_devices()?;
        return Ok(());
    }

    println!("=======================================================");
    println!("   🎙️ Linux AI Voice Changer: Mic -> Model -> Speaker  ");
    println!("=======================================================");

    // Register clean signal handler for SIGINT/SIGTERM
    ctrlc::set_handler(move || {
        println!("\nTermination signal received. Shutting down cleanly...");
        set_shutdown();
    }).expect("Error setting Ctrl-C handler");

    // 2. Validate model path or pick first available
    let initial_model = if args.model.exists() {
        args.model
    } else {
        let models = VoiceChangerApp::scan_models_dir();
        if let Some(first) = models.into_iter().next() {
            println!("Default model not found, auto-selecting: {:?}", first);
            first
        } else {
            eprintln!("Warning: No ONNX models found in models/ directory.");
            args.model
        }
    };

    // 3. Shared Controls & SPSC Ring Buffers
    let controls = Arc::new(AudioControls::default());
    controls.set_input_gain(args.gain);
    controls.set_output_volume(args.volume);

    // 65536 samples = ~4 seconds of buffer headroom — prevents overflow during scheduler hiccups
    // This must be much larger than DEFAULT_CHUNK_SIZE to avoid starving the inference thread.
    let (in_producer, mut in_consumer) = RingBuffer::<f32>::new(65536);
    let (mut out_producer, out_consumer) = RingBuffer::<f32>::new(65536);

    // Message type from UI to inference worker
    // Channel for commands from UI or CLI
    let (command_sender, command_receiver): (Sender<WorkerCommand>, Receiver<WorkerCommand>) = channel();
    // 4. Spawn Inference Worker Thread
    let worker_controls = Arc::clone(&controls);
    let worker_model = initial_model.clone();
    let worker_handle = thread::spawn(move || {
        println!("Inference worker thread started.");
        let default_cores = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
        let mut engine_opt = if worker_model.exists() {
            match VoiceChangerEngine::new(&worker_model, DEFAULT_HISTORY_SIZE, DEFAULT_CHUNK_SIZE, default_cores, ExecutionBackend::Cpu, 0.0) {
                Ok(engine) => {
                    println!("Loaded model: {:?}", worker_model);
                    Some(engine)
                }
                Err(e) => {
                    eprintln!("Error loading model {:?}: {}", worker_model, e);
                    None
                }
            }
        } else {
            None
        };

        let mut active_chunk_size = DEFAULT_CHUNK_SIZE;
        let mut chunk_buf = Vec::with_capacity(active_chunk_size);

        while !SHUTDOWN.load(Ordering::Relaxed) {
            // Check for hot-swap or config updates
            if let Ok(cmd) = command_receiver.try_recv() {
                match cmd {
                    WorkerCommand::SwapModel(path) => {
                        println!("Hot-swapping model to: {:?}", path);
                        if let Some(ref mut engine) = engine_opt {
                            if let Err(e) = engine.swap_model(&path) {
                                eprintln!("Failed to swap model: {}", e);
                            } else {
                                println!("Model successfully swapped!");
                            }
                        } else {
                            match VoiceChangerEngine::new(&path, DEFAULT_HISTORY_SIZE, active_chunk_size, default_cores, ExecutionBackend::Cpu, 0.0) {
                                Ok(engine) => {
                                    engine_opt = Some(engine);
                                    println!("Model successfully loaded!");
                                }
                                Err(e) => eprintln!("Failed to initialize engine with {:?}: {}", path, e),
                            }
                        }
                    }
                    WorkerCommand::ReinitEngine => {
                        println!("Re-initializing engine...");
                        if let Some(ref mut engine) = engine_opt {
                            if let Err(e) = engine.force_reinit() {
                                eprintln!("Failed to re-initialize engine: {}", e);
                            } else {
                                println!("Engine successfully re-initialized!");
                            }
                        }
                    }
                    WorkerCommand::UpdateSettings { chunk_size, history_size, num_cores, backend, pitch_semitones } => {
                        println!("Updating settings: chunk={}, history={}, cores={}", chunk_size, history_size, num_cores);
                        active_chunk_size = chunk_size;
                        if let Some(ref mut engine) = engine_opt {
                            if let Err(e) = engine.update_settings(history_size, chunk_size, num_cores, backend, pitch_semitones) {
                                eprintln!("Failed to update settings: {}", e);
                            } else {
                                println!("Settings successfully updated!");
                            }
                        }
                    }
                }
            }

            // Read samples from input ring buffer
            while chunk_buf.len() < active_chunk_size {
                if let Ok(sample) = in_consumer.pop() {
                    chunk_buf.push(sample);
                } else {
                    break;
                }
            }

            // If we have a full chunk, process it
            if chunk_buf.len() == active_chunk_size {
                let enabled = worker_controls.is_enabled();

                if enabled && engine_opt.is_some() {
                    let engine = engine_opt.as_mut().unwrap();
                    match engine.process_chunk(&chunk_buf) {
                        Ok(converted) => {
                            for sample in converted {
                                let _ = out_producer.push(sample);
                            }
                            worker_controls.set_rtf(engine.last_rtf());
                            worker_controls.set_latency_ms(engine.last_inference_time_ms());
                        }
                        Err(e) => {
                            eprintln!("Inference error: {}", e);
                            // Fallback to passthrough
                            for sample in &chunk_buf {
                                let _ = out_producer.push(*sample);
                            }
                        }
                    }
                } else {
                    // Bypass mode: raw mic audio passthrough
                    for sample in &chunk_buf {
                        let _ = out_producer.push(*sample);
                    }
                    worker_controls.set_rtf(0.0);
                    worker_controls.set_latency_ms(0.0);
                }

                chunk_buf.clear();
            } else {
                // Yield to the OS scheduler briefly while waiting for more mic samples.
                // spin_loop hint keeps the thread hot so it can respond immediately when
                // audio arrives — no artificial 2ms latency from thread::sleep.
                std::hint::spin_loop();
            }
        }

        println!("Inference worker thread stopped.");
    });

    // 5. Initialize Audio System (Default Mic -> Default Speaker)
    println!("Initializing direct audio pipeline...");
    let audio_system = AudioSystem::init(
        in_producer,
        out_consumer,
        Arc::clone(&controls),
        args.input_device.as_deref(),
        args.output_device.as_deref(),
    )?;

    println!("Audio pipeline is ACTIVE: speak into your mic to hear changed voice on speaker/headphones.\n");

    // 6. Run GUI or Headless Mode
    if args.headless {
        println!("Running in HEADLESS mode. Press Ctrl+C to stop.");
        while !shutdown_requested() {
            thread::sleep(Duration::from_millis(500));
            let rtf = controls.rtf();
            let lat = controls.latency_ms();
            let in_rms = controls.input_rms();
            let out_rms = controls.output_rms();
            let mode = if controls.is_enabled() { "ACTIVE" } else { "BYPASS" };
            print!("\r[{}] In RMS: {:.3} | Out RMS: {:.3} | RTF: {:.2}x | Latency: {:.1}ms   ", 
                mode, in_rms, out_rms, rtf, lat);
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        println!();
    } else {
        println!("Launching GUI control window...");
        let native_options = eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default()
                .with_inner_size([460.0, 520.0])
                .with_min_inner_size([400.0, 460.0])
                .with_title("Linux AI Voice Changer"),
            ..Default::default()
        };

        let app_controls = Arc::clone(&controls);
        let app_sender = command_sender.clone();
        let app_model = initial_model.clone();
        let in_name = audio_system.input_device_name.clone();
        let out_name = audio_system.output_device_name.clone();

        eframe::run_native(
            "Linux AI Voice Changer",
            native_options,
            Box::new(move |_cc| {
                Box::new(VoiceChangerApp::new(
                    app_controls,
                    app_sender,
                    app_model,
                    in_name,
                    out_name,
                ))
            }),
        ).map_err(|e| anyhow::anyhow!("GUI execution failed: {}", e))?;
    }

    set_shutdown();
    let _ = worker_handle.join();
    println!("Shutdown complete.");
    Ok(())
}

static SHUTDOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn shutdown_requested() -> bool {
    SHUTDOWN.load(Ordering::Relaxed)
}

fn set_shutdown() {
    SHUTDOWN.store(true, Ordering::Relaxed);
}

fn list_audio_devices() -> Result<()> {
    let host = cpal::default_host();
    println!("=== Audio Input Devices (Microphones) ===");
    for dev in host.input_devices()? {
        println!("  - {}", dev.name().unwrap_or_else(|_| "Unknown".into()));
    }

    println!("\n=== Audio Output Devices (Speakers / Headphones) ===");
    for dev in host.output_devices()? {
        println!("  - {}", dev.name().unwrap_or_else(|_| "Unknown".into()));
    }
    Ok(())
}
