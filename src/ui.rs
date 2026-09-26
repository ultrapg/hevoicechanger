use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use eframe::egui::{self, Color32, ProgressBar, RichText, Slider};
use crate::audio::AudioControls;

pub struct VoiceChangerApp {
    controls: Arc<AudioControls>,
    command_sender: Sender<crate::engine::WorkerCommand>,
    available_models: Vec<PathBuf>,
    selected_model_idx: usize,
    input_gain: f32,
    output_volume: f32,
    input_device_name: String,
    output_device_name: String,
    
    // Performance Settings
    chunk_size: usize,
    history_size: usize,
    num_cores: usize,
    backend: crate::engine::ExecutionBackend,
    pitch_semitones: f32,
    input_noise_gate: f32,
    output_noise_gate: f32,
}

impl VoiceChangerApp {
    pub fn new(
        controls: Arc<AudioControls>,
        command_sender: Sender<crate::engine::WorkerCommand>,
        initial_model: PathBuf,
        input_device_name: String,
        output_device_name: String,
    ) -> Self {
        let available_models = Self::scan_models_dir();
        let selected_model_idx = available_models
            .iter()
            .position(|p| p == &initial_model)
            .unwrap_or(0);

        let input_gain = controls.input_gain();
        let output_volume = controls.output_volume();
        let input_noise_gate = controls.input_noise_gate();
        let output_noise_gate = controls.output_noise_gate();
        let num_cores = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);

        Self {
            controls,
            command_sender,
            available_models,
            selected_model_idx,
            input_gain,
            output_volume,
            input_device_name,
            output_device_name,
            chunk_size: crate::engine::DEFAULT_CHUNK_SIZE,
            history_size: crate::engine::DEFAULT_HISTORY_SIZE,
            num_cores,
            backend: crate::engine::ExecutionBackend::Cpu,
            pitch_semitones: 0.0,
            input_noise_gate,
            output_noise_gate,
        }
    }

    pub fn scan_models_dir() -> Vec<PathBuf> {
        let mut models = Vec::new();
        if let Ok(entries) = std::fs::read_dir("models") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                    models.push(path);
                }
            }
        }
        models.sort();
        models
    }
}

impl eframe::App for VoiceChangerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(RichText::new("Linux AI Voice Changer").size(22.0).strong());
            ui.label(RichText::new("Direct Real-Time Processing: Mic -> AI Model -> Speakers").size(12.0).weak());
            ui.separator();

            // 1. Voice Model Selector
            ui.group(|ui| {
                ui.label(RichText::new("Active Voice Model").strong());
                if self.available_models.is_empty() {
                    ui.colored_label(Color32::RED, "No .onnx models found in ./models directory!");
                } else {
                    let current_name = self.available_models[self.selected_model_idx]
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("Unknown");

                    egui::ComboBox::from_label("Voice Profile")
                        .selected_text(current_name)
                        .show_ui(ui, |ui| {
                            for (idx, path) in self.available_models.iter().enumerate() {
                                let name = path.file_name()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("Unknown");

                                if ui.selectable_label(self.selected_model_idx == idx, name).clicked() {
                                    if self.selected_model_idx != idx {
                                        self.selected_model_idx = idx;
                                        let _ = self.command_sender.send(crate::engine::WorkerCommand::SwapModel(path.clone()));
                                    }
                                }
                            }
                        });
                }

                if ui.button("Refresh Models").clicked() {
                    self.available_models = Self::scan_models_dir();
                }
            });

            ui.add_space(8.0);

            // 2. Main On/Off Toggle
            ui.group(|ui| {
                let mut enabled = self.controls.is_enabled();
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Mode:").strong());
                    if enabled {
                        ui.colored_label(Color32::GREEN, "ACTIVE (VOICE CONVERTED)");
                    } else {
                        ui.colored_label(Color32::YELLOW, "BYPASS (RAW MIC)");
                    }
                });

                let button_text = if enabled {
                    RichText::new("Switch to Bypass (Raw Mic)").color(Color32::WHITE)
                } else {
                    RichText::new("Enable Voice Changer").color(Color32::WHITE)
                };

                if ui.button(button_text).clicked() {
                    enabled = !enabled;
                    self.controls.set_enabled(enabled);
                }
            });

            ui.add_space(8.0);

            // 3. Audio Levels & Sliders
            ui.group(|ui| {
                ui.label(RichText::new("Audio Levels & Gain").strong());

                ui.horizontal(|ui| {
                    ui.label("Input Gain:");
                    if ui.add(Slider::new(&mut self.input_gain, 0.0..=3.0).text("x")).changed() {
                        self.controls.set_input_gain(self.input_gain);
                    }
                });

                let in_rms = self.controls.input_rms();
                let in_progress = (in_rms * 5.0).clamp(0.0, 1.0);
                ui.horizontal(|ui| {
                    ui.label("Mic In RMS:");
                    ui.add(ProgressBar::new(in_progress).show_percentage());
                });

                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Output Vol:");
                    if ui.add(Slider::new(&mut self.output_volume, 0.0..=10.0).text("x")).changed() {
                        self.controls.set_output_volume(self.output_volume);
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Pitch Shift:");
                    if ui.add(Slider::new(&mut self.pitch_semitones, -12.0..=12.0).text("Semitones")).changed() {
                        let _ = self.command_sender.send(crate::engine::WorkerCommand::UpdateSettings {
                            chunk_size: self.chunk_size,
                            history_size: self.history_size,
                            num_cores: self.num_cores,
                            backend: self.backend,
                        pitch_semitones: self.pitch_semitones,
                        });
                    }
                });

                let out_rms = self.controls.output_rms();
                let out_progress = (out_rms * 5.0).clamp(0.0, 1.0);
                ui.horizontal(|ui| {
                    ui.label("Speaker RMS:");
                    ui.add(ProgressBar::new(out_progress).show_percentage());
                });
            });

            ui.add_space(8.0);

            // 4. Advanced Performance Settings
            ui.group(|ui| {
                ui.label(RichText::new("Advanced Performance Settings").strong());
                
                if ui.button("Force Reboot AI Engine (Fix Glitches)").clicked() {
                    let _ = self.command_sender.send(crate::engine::WorkerCommand::ReinitEngine);
                }
                ui.add_space(4.0);
                let mut settings_changed = false;


                ui.horizontal(|ui| {
                    ui.label("CPU Cores (Intra):");
                    let max_cores = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
                    if ui.add(Slider::new(&mut self.num_cores, 1..=max_cores).text("Cores")).changed() {
                        settings_changed = true;
                    }
                });
                
                ui.horizontal(|ui| {
                    ui.label("GPU Backend:");
                    let prev_backend = self.backend;
                    let selected_text = match self.backend {
                        crate::engine::ExecutionBackend::Cpu => "CPU (AVX2/SIMD)",
                        crate::engine::ExecutionBackend::WebGpu => "WebGPU (Vulkan)",
                    };
                    
                    egui::ComboBox::from_id_source("gpu_selector")
                        .selected_text(selected_text)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.backend, crate::engine::ExecutionBackend::Cpu, "CPU (AVX2/SIMD)");
                            ui.selectable_value(&mut self.backend, crate::engine::ExecutionBackend::WebGpu, "WebGPU (Vulkan)");
                        });
                        
                    if prev_backend != self.backend {
                        settings_changed = true;
                    }
                });

                if settings_changed {
                    let _ = self.command_sender.send(crate::engine::WorkerCommand::UpdateSettings {
                        chunk_size: self.chunk_size,
                        history_size: self.history_size,
                        num_cores: self.num_cores,
                        backend: self.backend,
                        pitch_semitones: self.pitch_semitones,
                    });
                }
                
                ui.separator();
                
                ui.horizontal(|ui| {
                    ui.label("Input Noise Reduction:");
                    if ui.add(Slider::new(&mut self.input_noise_gate, 0.0..=0.1).text("Thresh")).changed() {
                        self.controls.set_input_noise_gate(self.input_noise_gate);
                    }
                });
                
                ui.horizontal(|ui| {
                    ui.label("Output Noise Reduction:");
                    if ui.add(Slider::new(&mut self.output_noise_gate, 0.0..=0.1).text("Thresh")).changed() {
                        self.controls.set_output_noise_gate(self.output_noise_gate);
                    }
                });
            });

            ui.add_space(8.0);

            // 5. Performance & Diagnostics
            ui.group(|ui| {
                ui.label(RichText::new("Performance & Device Status").strong());

                let rtf = self.controls.rtf();
                let latency = self.controls.latency_ms();

                ui.horizontal(|ui| {
                    ui.label("Real-Time Factor (RTF):");
                    if rtf < 0.6 {
                        ui.colored_label(Color32::GREEN, format!("{:.3}x (Fast / No Lag)", rtf));
                    } else if rtf < 1.0 {
                        ui.colored_label(Color32::YELLOW, format!("{:.3}x (Acceptable)", rtf));
                    } else {
                        ui.colored_label(Color32::RED, format!("{:.3}x (Under-run risk)", rtf));
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Inference Latency:");
                    ui.label(format!("{:.2} ms", latency));
                });

                ui.horizontal(|ui| {
                    ui.label("Mic Device:");
                    ui.label(RichText::new(&self.input_device_name).color(Color32::LIGHT_BLUE));
                });

                ui.horizontal(|ui| {
                    ui.label("Speaker Device:");
                    ui.label(RichText::new(&self.output_device_name).color(Color32::LIGHT_BLUE));
                });
            });
        });
    }
}
