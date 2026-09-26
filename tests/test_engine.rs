use std::path::Path;
use hevoicechanger::engine::{VoiceChangerEngine, DEFAULT_CHUNK_SIZE, DEFAULT_HISTORY_SIZE, SAMPLE_RATE};

#[test]
fn test_engine_inference_and_rtf() {
    let model_path = Path::new("models/female_voice.onnx");
    assert!(model_path.exists(), "Model file models/female_voice.onnx must exist");

    let mut engine = VoiceChangerEngine::new(model_path, DEFAULT_HISTORY_SIZE, DEFAULT_CHUNK_SIZE)
        .expect("Failed to initialize engine");

    // Generate 1 second of synthetic 440Hz sine wave at 16kHz
    let duration_secs = 0.5;
    let total_samples = (SAMPLE_RATE as f32 * duration_secs) as usize;
    let mut input_audio = Vec::with_capacity(total_samples);
    for i in 0..total_samples {
        let t = i as f32 / SAMPLE_RATE as f32;
        let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        input_audio.push(sample);
    }

    // Process in chunks of 256 samples
    let mut output_audio = Vec::new();
    let mut total_rtf = 0.0;
    let mut chunk_count = 0;

    for chunk in input_audio.chunks(DEFAULT_CHUNK_SIZE) {
        if chunk.len() == DEFAULT_CHUNK_SIZE {
            let converted = engine.process_chunk(chunk).expect("Failed to process chunk");
            assert_eq!(converted.len(), DEFAULT_CHUNK_SIZE);
            total_rtf += engine.last_rtf();
            chunk_count += 1;
            output_audio.extend_from_slice(&converted);
        }
    }

    let avg_rtf = total_rtf / chunk_count as f32;
    println!("Processed {} chunks, Avg RTF: {:.3}, Last latency: {:.2} ms", 
        chunk_count, avg_rtf, engine.last_inference_time_ms());

    // Check that audio output is non-empty and has reasonable signal
    let rms: f32 = (output_audio.iter().map(|s| s * s).sum::<f32>() / output_audio.len() as f32).sqrt();
    println!("Output RMS: {:.4}", rms);
    assert!(rms > 0.001, "Output audio should have non-zero signal");

    // Test model hot-swapping to female_voice_hfg.onnx
    let hfg_path = Path::new("models/female_voice_hfg.onnx");
    if hfg_path.exists() {
        engine.swap_model(hfg_path).expect("Failed to swap model to HFG");
        let sample_chunk = vec![0.1; DEFAULT_CHUNK_SIZE];
        let converted = engine.process_chunk(&sample_chunk).expect("Failed to process chunk after swap");
        assert_eq!(converted.len(), DEFAULT_CHUNK_SIZE);
        println!("Successfully verified model hot-swapping to HFG variant!");
    }
}
