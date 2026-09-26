use std::path::{Path, PathBuf};
use std::time::Instant;
use anyhow::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

/// Receptive field history fed to the causal CNN each inference call.
/// 16000 samples = 1 second of history at 16kHz — significantly more context
/// to ensure the ONNX model's internal zero-state initialization doesn't corrupt the output.
pub const DEFAULT_HISTORY_SIZE: usize = 16000;

/// Chunk size (samples processed per inference call).
/// 2048 samples @ 16kHz = 128ms per chunk.
/// Larger chunks → fewer ONNX calls per second → better CPU vectorisation
/// and less overhead, so all 8 cores saturate instead of spinning idle.
pub const DEFAULT_CHUNK_SIZE: usize = 2048;

pub const SAMPLE_RATE: u32 = 16000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionBackend {
    Cpu,
    WebGpu,
}

#[derive(Debug)]
pub enum WorkerCommand {
    SwapModel(PathBuf),
    ReinitEngine,
    UpdateSettings {
        chunk_size: usize,
        history_size: usize,
        num_cores: usize,
        backend: ExecutionBackend,
        pitch_semitones: f32,
    }
}

pub struct VoiceChangerEngine {
    session: Session,
    model_path: PathBuf,
    pub history: Vec<f32>,
    pub history_size: usize,
    pub chunk_size: usize,
    pub num_cores: usize,
    pub pitch_semitones: f32,
    pitch_shifter: Box<pitch_shift::Shifter<Box<[f32; pitch_shift::TOTAL_F32]>>>,
    pub backend: ExecutionBackend,
    // Metrics
    last_inference_time_ms: f32,
    last_rtf: f32,
}

impl VoiceChangerEngine {
    pub fn new<P: AsRef<Path>>(model_path: P, history_size: usize, chunk_size: usize, num_cores: usize, backend: ExecutionBackend, pitch_semitones: f32) -> Result<Self> {
        let path_buf = model_path.as_ref().to_path_buf();
        let session = Self::load_session(&path_buf, num_cores, backend)?;

        let state_vec = vec![0.0; pitch_shift::TOTAL_F32];
        let state_box: Box<[f32; pitch_shift::TOTAL_F32]> = state_vec.try_into().unwrap();
        let pitch_shifter = Box::new(pitch_shift::Shifter::new(state_box));

        Ok(Self {
            session,
            model_path: path_buf,
            history: vec![0.0; history_size],
            history_size,
            chunk_size,
            num_cores,
            pitch_semitones,
            pitch_shifter,
            backend,
            last_inference_time_ms: 0.0,
            last_rtf: 0.0,
        })
    }

    pub fn load_session(model_path: &Path, num_cores: usize, backend: ExecutionBackend) -> Result<Session> {
        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create Session builder: {e:?}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {e:?}"))?
            .with_intra_threads(num_cores as _)
            .map_err(|e| anyhow::anyhow!("Failed to set intra threads: {e:?}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("Failed to set inter threads: {e:?}"))?;

        match backend {
            ExecutionBackend::WebGpu => {
                builder = builder.with_execution_providers([ort::ep::WebGPU::default().build()])
                    .map_err(|e| anyhow::anyhow!("Failed to set WebGPU Execution Provider: {e:?}"))?;
            }
            ExecutionBackend::Cpu => {} // CPU is default
        }
        let session = builder.commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load ONNX model from {model_path:?}: {e:?}"))?;

        Ok(session)
    }

    pub fn force_reinit(&mut self) -> Result<()> {
        let new_session = Self::load_session(&self.model_path, self.num_cores, self.backend)?;
        self.session = new_session;
        self.history.fill(0.0);
        Ok(())
    }

    pub fn swap_model<P: AsRef<Path>>(&mut self, model_path: P) -> Result<()> {
        let path_buf = model_path.as_ref().to_path_buf();
        let new_session = Self::load_session(&path_buf, self.num_cores, self.backend)?;
        self.session = new_session;
        self.model_path = path_buf;
        self.history.fill(0.0);
        Ok(())
    }

    pub fn update_settings(&mut self, history_size: usize, chunk_size: usize, num_cores: usize, backend: ExecutionBackend, pitch_semitones: f32) -> Result<()> {
        self.pitch_semitones = pitch_semitones;
        self.chunk_size = chunk_size;
        
        if self.history_size != history_size {
            self.history_size = history_size;
            self.history = vec![0.0; history_size];
        }

        if self.num_cores != num_cores || self.backend != backend {
            self.num_cores = num_cores;
            self.backend = backend;
            let new_session = Self::load_session(&self.model_path, num_cores, backend)?;
            self.session = new_session;
        }

        Ok(())
    }

    pub fn reset_history(&mut self) {
        self.history.fill(0.0);
    }

    pub fn process_chunk(&mut self, incoming_chunk: &[f32]) -> Result<Vec<f32>> {
        let start_time = Instant::now();

        // 1. Combine historical context with incoming chunk
        let total_samples = self.history_size + incoming_chunk.len();
        let mut full_input = Vec::with_capacity(total_samples);
        full_input.extend_from_slice(&self.history);
        full_input.extend_from_slice(incoming_chunk);

        // 2. Wrap as Tensor with shape [1, 1, total_samples]
        let input_tensor = Tensor::from_array(([1, 1, total_samples], full_input))
            .map_err(|e| anyhow::anyhow!("Failed to create ORT tensor: {e:?}"))?;

        // 3. Run inference
        let outputs = self.session.run(ort::inputs!["audio_in" => input_tensor])
            .map_err(|e| anyhow::anyhow!("Inference execution failed: {e:?}"))?;

        let (_shape, output_slice) = outputs["audio_out"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract output tensor: {e:?}"))?;

        // 4. Update history for the next iteration (slide window)
        let incoming_len = incoming_chunk.len();
        if incoming_len >= self.history_size {
            self.history.copy_from_slice(&incoming_chunk[incoming_len - self.history_size..]);
        } else {
            self.history.drain(0..incoming_len);
            self.history.extend_from_slice(incoming_chunk);
        }

        // 5. Discard receptive field context and keep only converted chunk
        let converted = if output_slice.len() > self.history_size {
            output_slice[self.history_size..].to_vec()
        } else {
            output_slice.to_vec()
        };

        // 6. Update metrics
        let elapsed = start_time.elapsed().as_secs_f32();
        self.last_inference_time_ms = elapsed * 1000.0;
        let chunk_duration = incoming_chunk.len() as f32 / SAMPLE_RATE as f32;
        self.last_rtf = elapsed / chunk_duration;

        let mut final_output = Vec::with_capacity(converted.len());
        if self.pitch_semitones.abs() > 0.01 {
            for chunk in converted.chunks(128) {
                if chunk.len() == 128 {
                    let shifted = self.pitch_shifter.shift(chunk, self.pitch_semitones, 128, SAMPLE_RATE as f32);
                    final_output.extend_from_slice(shifted);
                } else {
                    final_output.extend_from_slice(chunk);
                }
            }
        } else {
            final_output = converted;
        }

        Ok(final_output)
    }

    pub fn last_inference_time_ms(&self) -> f32 {
        self.last_inference_time_ms
    }

    pub fn last_rtf(&self) -> f32 {
        self.last_rtf
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }
}
