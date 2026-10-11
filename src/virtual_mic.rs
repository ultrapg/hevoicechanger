use std::io::{BufRead, BufReader, Write};
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub const DEFAULT_SINK_NAME: &str = "HEVoiceSink";
pub const DEFAULT_SINK_DESC: &str = "HEVoicechanger Virtual Output";
pub const DEFAULT_MIC_NAME: &str = "HEVoiceMic";
pub const DEFAULT_MIC_DESC: &str = "HEVoicechanger Virtual Microphone";

const STARTUP_TIMEOUT: Duration = Duration::from_secs(5);
const LINK_TIMEOUT: Duration = Duration::from_millis(2000);
const POLL_INTERVAL: Duration = Duration::from_millis(50);
const SHUTDOWN_GRACE: Duration = Duration::from_millis(1000);
const REMOVAL_TIMEOUT: Duration = Duration::from_millis(1000);

#[derive(Debug, Clone)]
pub struct VirtualMicConfig {
    pub speaker_name: String,
    pub speaker_desc: String,
    pub mic_name: String,
    pub mic_desc: String,
    pub channels: u32,
    pub latency_ms: Option<u32>,
}

impl Default for VirtualMicConfig {
    fn default() -> Self {
        Self {
            speaker_name: DEFAULT_SINK_NAME.to_string(),
            speaker_desc: DEFAULT_SINK_DESC.to_string(),
            mic_name: DEFAULT_MIC_NAME.to_string(),
            mic_desc: DEFAULT_MIC_DESC.to_string(),
            channels: 2,
            latency_ms: None,
        }
    }
}

pub enum BackendHandle {
    PipeWire {
        child: Child,
        _stdin: ChildStdin,
        _last_stderr: Arc<Mutex<Vec<String>>>,
    },
    PulseAudio {
        null_sink_id: u32,
        remap_source_id: u32,
    },
}

pub struct VirtualMicBridge {
    pub config: VirtualMicConfig,
    backend_handle: Option<BackendHandle>,
    pub active_backend_name: String,
    pub is_active: bool,
    pub monitor_enabled: bool,
    discovered_monitor_ports: Vec<String>,
    pulse_loopback_id: Option<u32>,
    sink_id: Option<u32>,
    source_id: Option<u32>,
}

impl VirtualMicConfig {
    pub fn is_valid_name(name: &str) -> bool {
        !name.is_empty()
            && name.len() <= 128
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    }
}

impl VirtualMicBridge {
    pub fn new(config: VirtualMicConfig) -> Self {
        Self {
            config,
            backend_handle: None,
            active_backend_name: String::new(),
            is_active: false,
            monitor_enabled: false,
            discovered_monitor_ports: Vec::new(),
            pulse_loopback_id: None,
            sink_id: None,
            source_id: None,
        }
    }

    /// Check if PipeWire CLI (pw-cli) is available
    pub fn is_pipewire_available() -> bool {
        Command::new("pw-cli")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Check if PulseAudio CLI (pactl) is available
    pub fn is_pulse_available() -> bool {
        Command::new("pactl")
            .arg("info")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Clean up any existing virtual devices left behind from previous sessions
    pub fn cleanup_existing(speaker_name: &str, mic_name: &str) {
        // Destroy PipeWire nodes
        cleanup_pipewire_nodes(speaker_name, mic_name);

        // Unload PulseAudio modules matching our names
        if let Ok(modules) = list_pulse_modules() {
            for (id, name, args) in modules {
                let matches = match name.as_str() {
                    "module-null-sink" => has_module_arg(&args, "sink_name", speaker_name),
                    "module-remap-source" => {
                        has_module_arg(&args, "source_name", mic_name)
                            || has_module_arg(&args, "source_name", speaker_name)
                    }
                    _ => false,
                };
                if matches {
                    let _ = unload_pulse_module(id);
                }
            }
        }
    }

    /// Start the virtual microphone bridge
    pub fn start(&mut self) -> Result<(), String> {
        Self::cleanup_existing(&self.config.speaker_name, &self.config.mic_name);

        let try_pipewire = Self::is_pipewire_available();
        let try_pulse = Self::is_pulse_available();

        if !try_pipewire && !try_pulse {
            return Err("Neither PipeWire (pw-cli) nor PulseAudio (pactl) is available on this system".to_string());
        }

        let mut last_err = String::new();

        if try_pipewire {
            match self.setup_pipewire() {
                Ok(()) => {
                    self.active_backend_name = "PipeWire".to_string();
                    self.is_active = true;
                    self.resolve_device_ids();
                    return Ok(());
                }
                Err(e) => {
                    self.stop();
                    last_err = e;
                }
            }
        }

        if try_pulse {
            match self.setup_pulse() {
                Ok(()) => {
                    self.active_backend_name = "PulseAudio".to_string();
                    self.is_active = true;
                    self.resolve_device_ids();
                    return Ok(());
                }
                Err(e) => {
                    self.stop();
                    last_err = e;
                }
            }
        }

        Err(format!("Failed to initialize virtual audio bridge: {last_err}"))
    }

    fn resolve_device_ids(&mut self) {
        self.sink_id = find_sink(&self.config.speaker_name).ok().flatten().map(|s| s.id);
        self.source_id = find_source(&self.config.mic_name).ok().flatten().map(|s| s.id);
    }

    /// Set up native PipeWire virtual devices via pw-cli
    fn setup_pipewire(&mut self) -> Result<(), String> {
        let pos = match self.config.channels {
            1 => "[ MONO ]",
            _ => "[ FL FR ]",
        };

        let latency = match self.config.latency_ms {
            Some(ms) if ms > 0 => format!(" node.latency = \"{ms}/48000\""),
            _ => String::new(),
        };

        let mut cmd = Command::new("pw-cli");
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::piped());

        // Linux kernel fail-safe: deliver SIGTERM to pw-cli if parent dies
        let parent = std::process::id() as libc::pid_t;
        unsafe {
            cmd.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    libc::_exit(1);
                }
                Ok(())
            });
        }

        let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn pw-cli: {e}"))?;

        let mut stdin = match child.stdin.take() {
            Some(stdin) => stdin,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("Failed to acquire pw-cli stdin".to_string());
            }
        };

        let last_stderr = Arc::new(Mutex::new(Vec::new()));
        if let Some(stderr) = child.stderr.take() {
            let stderr_storage = Arc::clone(&last_stderr);
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    let mut lines = stderr_storage.lock().unwrap();
                    if lines.len() >= 20 {
                        lines.remove(0);
                    }
                    lines.push(line);
                }
            });
        }

        let create_speaker = format!(
            "create-node adapter {{ factory.name = support.null-audio-sink node.name = \"{}\" node.description = \"{}\" media.class = \"Audio/Sink\" audio.position = {}{} }}\n",
            self.config.speaker_name, self.config.speaker_desc, pos, latency
        );
        let create_mic = format!(
            "create-node adapter {{ factory.name = support.null-audio-sink node.name = \"{}\" node.description = \"{}\" media.class = \"Audio/Source/Virtual\" audio.position = {}{} }}\n",
            self.config.mic_name, self.config.mic_desc, pos, latency
        );

        let written = stdin
            .write_all(create_speaker.as_bytes())
            .and_then(|()| stdin.write_all(create_mic.as_bytes()))
            .and_then(|()| stdin.flush());

        if let Err(e) = written {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("Failed to send create-node commands to pw-cli: {e}"));
        }

        self.backend_handle = Some(BackendHandle::PipeWire {
            child,
            _stdin: stdin,
            _last_stderr: last_stderr,
        });

        // Wait for nodes to appear in PipeWire
        self.wait_for_devices(true)?;

        // Link HEVoiceSink:monitor_* to HEVoiceMic:input_*
        self.link_pipewire_bridge()?;

        Ok(())
    }

    /// Set up PulseAudio null-sink and remap-source modules
    fn setup_pulse(&mut self) -> Result<(), String> {
        let null_sink_cmd = Command::new("pactl")
            .args([
                "load-module",
                "module-null-sink",
                &format!("sink_name={}", self.config.speaker_name),
                &format!(
                    "sink_properties=device.description=\"{}\"",
                    self.config.speaker_desc
                ),
                &format!("channels={}", self.config.channels),
            ])
            .output()
            .map_err(|e| format!("Failed to run pactl: {e}"))?;

        if !null_sink_cmd.status.success() {
            return Err(format!(
                "Failed to create null-sink: {}",
                String::from_utf8_lossy(&null_sink_cmd.stderr).trim()
            ));
        }

        let null_sink_id = String::from_utf8_lossy(&null_sink_cmd.stdout)
            .trim()
            .parse::<u32>()
            .map_err(|e| format!("Failed to parse null-sink module ID: {e}"))?;

        let remap_cmd = Command::new("pactl")
            .args([
                "load-module",
                "module-remap-source",
                &format!("source_name={}", self.config.mic_name),
                &format!("master={}.monitor", self.config.speaker_name),
                &format!(
                    "source_properties=device.description=\"{}\"",
                    self.config.mic_desc
                ),
                &format!("channels={}", self.config.channels),
            ])
            .output()
            .map_err(|e| format!("Failed to run pactl: {e}"))?;

        if !remap_cmd.status.success() {
            let _ = unload_pulse_module(null_sink_id);
            return Err(format!(
                "Failed to create remap-source: {}",
                String::from_utf8_lossy(&remap_cmd.stderr).trim()
            ));
        }

        let remap_source_id = String::from_utf8_lossy(&remap_cmd.stdout)
            .trim()
            .parse::<u32>()
            .map_err(|e| {
                let _ = unload_pulse_module(null_sink_id);
                format!("Failed to parse remap-source module ID: {e}")
            })?;

        self.backend_handle = Some(BackendHandle::PulseAudio {
            null_sink_id,
            remap_source_id,
        });

        self.wait_for_devices(false)?;

        Ok(())
    }

    fn wait_for_devices(&self, use_pipewire: bool) -> Result<(), String> {
        let start = Instant::now();
        loop {
            let sink_ok = if use_pipewire {
                pw_node_exists(&self.config.speaker_name)
            } else {
                find_sink(&self.config.speaker_name).map(|d| d.is_some()).unwrap_or(false)
            };

            let source_ok = if use_pipewire {
                pw_node_exists(&self.config.mic_name)
            } else {
                find_source(&self.config.mic_name).map(|d| d.is_some()).unwrap_or(false)
            };

            if sink_ok && source_ok {
                return Ok(());
            }

            if start.elapsed() >= STARTUP_TIMEOUT {
                return Err(format!(
                    "Timed out waiting for virtual audio devices (Sink: {}, Mic: {})",
                    if sink_ok { "OK" } else { "Missing" },
                    if source_ok { "OK" } else { "Missing" }
                ));
            }

            thread::sleep(POLL_INTERVAL);
        }
    }

    fn link_pipewire_bridge(&self) -> Result<(), String> {
        let speaker = &self.config.speaker_name;
        let mic = &self.config.mic_name;
        let pairs = match self.config.channels {
            1 => vec![(
                format!("{speaker}:monitor_MONO"),
                format!("{mic}:input_MONO"),
            )],
            _ => vec![
                (format!("{speaker}:monitor_FL"), format!("{mic}:input_FL")),
                (format!("{speaker}:monitor_FR"), format!("{mic}:input_FR")),
            ],
        };

        let deadline = Instant::now() + LINK_TIMEOUT;
        loop {
            let failures: Vec<String> = pairs
                .iter()
                .filter_map(|(out_p, in_p)| link_ports(out_p, in_p).err())
                .collect();

            if failures.is_empty() {
                return Ok(());
            }

            if Instant::now() >= deadline {
                return Err(format!(
                    "Failed to link virtual speaker to virtual mic: {}",
                    failures.join("; ")
                ));
            }

            thread::sleep(POLL_INTERVAL);
        }
    }

    /// Attach hevoicechanger's playback stream to the Virtual Sink and configure monitoring
    pub fn attach_hevoicechanger(&mut self, monitor: bool) {
        self.monitor_enabled = monitor;

        if let Some(BackendHandle::PipeWire { .. }) = &self.backend_handle {
            // Find hevoicechanger playback ports in PipeWire
            let deadline = Instant::now() + Duration::from_millis(3000);
            let mut hvc_out_ports = Vec::new();

            while Instant::now() < deadline {
                hvc_out_ports = find_hevoicechanger_output_ports();
                if !hvc_out_ports.is_empty() {
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }

            if hvc_out_ports.is_empty() {
                eprintln!("[!] Could not locate hevoicechanger audio output ports in PipeWire");
                return;
            }

            // Detect what physical output ports hevoicechanger was initially connected to (for monitor toggle)
            let existing_links = get_connected_input_ports_for(&hvc_out_ports);
            self.discovered_monitor_ports = existing_links
                .into_iter()
                .filter(|p| !p.starts_with(&self.config.speaker_name))
                .collect();

            // Link hevoicechanger output ports to HEVoiceSink:playback_*
            let sink = &self.config.speaker_name;
            for port in &hvc_out_ports {
                if port.ends_with("_FL") || port.ends_with("_0") || port.ends_with("_MONO") {
                    let _ = link_ports(port, &format!("{sink}:playback_FL"));
                    let _ = link_ports(port, &format!("{sink}:playback_MONO"));
                }
                if port.ends_with("_FR") || port.ends_with("_1") {
                    let _ = link_ports(port, &format!("{sink}:playback_FR"));
                }
            }

            // If monitor is false, disconnect from physical headphones to prevent echo
            if !monitor {
                for hvc_port in &hvc_out_ports {
                    for phys_port in &self.discovered_monitor_ports {
                        let _ = unlink_ports(hvc_port, phys_port);
                    }
                }
            }
        } else if let Some(BackendHandle::PulseAudio { .. }) = &self.backend_handle {
            // In PulseAudio, find the sink-input for hevoicechanger and move it to HEVoiceSink
            let deadline = Instant::now() + Duration::from_millis(3000);
            let mut stream_id = None;

            while Instant::now() < deadline {
                if let Ok(inputs) = list_sink_inputs_verbose() {
                    for (id, name) in inputs {
                        if name.contains("hevoicechanger") || name.contains("ALSA") {
                            stream_id = Some(id);
                            break;
                        }
                    }
                }
                if stream_id.is_some() {
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }

            if let Some(id) = stream_id {
                let _ = Command::new("pactl")
                    .args(["move-sink-input", &id.to_string(), &self.config.speaker_name])
                    .output();
            }

            if monitor {
                self.enable_pulse_loopback();
            }
        }
    }

    /// Toggle headphone monitoring (Hear Myself)
    pub fn set_monitor(&mut self, enable: bool) {
        if self.monitor_enabled == enable {
            return;
        }
        self.monitor_enabled = enable;

        if let Some(BackendHandle::PipeWire { .. }) = &self.backend_handle {
            let hvc_out_ports = find_hevoicechanger_output_ports();
            if enable {
                // Connect back to physical headphones
                for hvc_port in &hvc_out_ports {
                    for phys_port in &self.discovered_monitor_ports {
                        if (hvc_port.ends_with("_FL") && phys_port.ends_with("_FL"))
                            || (hvc_port.ends_with("_FR") && phys_port.ends_with("_FR"))
                            || (!hvc_port.ends_with("_FR") && !phys_port.ends_with("_FR"))
                        {
                            let _ = link_ports(hvc_port, phys_port);
                        }
                    }
                }
            } else {
                // Disconnect from physical headphones
                for hvc_port in &hvc_out_ports {
                    for phys_port in &self.discovered_monitor_ports {
                        let _ = unlink_ports(hvc_port, phys_port);
                    }
                }
            }
        } else if let Some(BackendHandle::PulseAudio { .. }) = &self.backend_handle {
            if enable {
                self.enable_pulse_loopback();
            } else {
                self.disable_pulse_loopback();
            }
        }
    }

    fn enable_pulse_loopback(&mut self) {
        if self.pulse_loopback_id.is_none() {
            let cmd = Command::new("pactl")
                .args([
                    "load-module",
                    "module-loopback",
                    &format!("source={}.monitor", self.config.speaker_name),
                    "latency_msec=20",
                ])
                .output();

            if let Ok(out) = cmd {
                if out.status.success() {
                    let id_str = String::from_utf8_lossy(&out.stdout);
                    if let Ok(id) = id_str.trim().parse::<u32>() {
                        self.pulse_loopback_id = Some(id);
                    }
                }
            }
        }
    }

    fn disable_pulse_loopback(&mut self) {
        if let Some(id) = self.pulse_loopback_id.take() {
            let _ = unload_pulse_module(id);
        }
    }

    /// Query how many apps are capturing the virtual microphone (e.g. Discord, OBS)
    pub fn active_listener_count(&self) -> usize {
        let Some(source_id) = self.source_id else {
            return 0;
        };

        list_source_outputs()
            .map(|outputs| {
                outputs
                    .into_iter()
                    .filter(|s| s.target_device_id == source_id)
                    .count()
            })
            .unwrap_or(0)
    }

    /// Cleanly stop the bridge and destroy virtual devices
    pub fn stop(&mut self) {
        self.is_active = false;
        self.disable_pulse_loopback();

        if let Some(handle) = self.backend_handle.take() {
            match handle {
                BackendHandle::PipeWire {
                    mut child, _stdin, ..
                } => {
                    drop(_stdin);
                    unsafe {
                        libc::kill(child.id() as libc::pid_t, libc::SIGTERM);
                    }

                    let start = Instant::now();
                    while start.elapsed() < SHUTDOWN_GRACE {
                        if let Ok(Some(_)) = child.try_wait() {
                            break;
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                    let _ = child.kill();
                    let _ = child.wait();
                }
                BackendHandle::PulseAudio {
                    null_sink_id,
                    remap_source_id,
                } => {
                    let _ = unload_pulse_module(remap_source_id);
                    let _ = unload_pulse_module(null_sink_id);
                }
            }
        }

        // Secondary sweep
        cleanup_pipewire_nodes(&self.config.speaker_name, &self.config.mic_name);

        let start = Instant::now();
        while start.elapsed() < REMOVAL_TIMEOUT {
            let sink_gone = find_sink(&self.config.speaker_name)
                .map(|s| s.is_none())
                .unwrap_or(true);
            let mic_gone = find_source(&self.config.mic_name)
                .map(|m| m.is_none())
                .unwrap_or(true);
            if sink_gone && mic_gone {
                break;
            }
            thread::sleep(Duration::from_millis(30));
        }
    }
}

impl Drop for VirtualMicBridge {
    fn drop(&mut self) {
        self.stop();
    }
}

// ============================================================================
// Low-level audio helpers adapted from vout2mic
// ============================================================================

#[derive(Debug, Clone)]
pub struct AudioDevice {
    pub id: u32,
    pub name: String,
    pub sample_spec: String,
    pub state: String,
}

#[derive(Debug, Clone)]
pub struct StreamInfo {
    pub target_device_id: u32,
}

fn pactl(args: &[&str]) -> Result<String, String> {
    let output = Command::new("pactl")
        .args(args)
        .output()
        .map_err(|e| format!("Failed to run `pactl {}`: {e}", args.join(" ")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("`pactl {}` failed: {}", args.join(" "), stderr.trim()));
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

pub fn list_sinks() -> Result<Vec<AudioDevice>, String> {
    Ok(parse_device_short(&pactl(&["list", "sinks", "short"])?))
}

pub fn list_sources() -> Result<Vec<AudioDevice>, String> {
    Ok(parse_device_short(&pactl(&["list", "sources", "short"])?))
}

fn parse_device_short(text: &str) -> Vec<AudioDevice> {
    text.lines()
        .filter_map(|line| {
            let mut parts = line.split('\t').map(str::trim);
            let id = parts.next()?.parse::<u32>().ok()?;
            let name = parts.next()?.to_string();
            let _driver = parts.next()?;
            let sample_spec = parts.next()?.to_string();
            let state = parts.next().unwrap_or_default().to_string();
            Some(AudioDevice {
                id,
                name,
                sample_spec,
                state,
            })
        })
        .collect()
}

pub fn find_sink(name: &str) -> Result<Option<AudioDevice>, String> {
    Ok(list_sinks()?.into_iter().find(|s| s.name == name))
}

pub fn find_source(name: &str) -> Result<Option<AudioDevice>, String> {
    Ok(list_sources()?.into_iter().find(|s| s.name == name))
}

pub fn list_source_outputs() -> Result<Vec<StreamInfo>, String> {
    let text = pactl(&["list", "source-outputs", "short"])?;
    Ok(text
        .lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let _stream_id = parts.next()?.parse::<u32>().ok()?;
            let target_device_id = parts.next()?.parse::<u32>().ok()?;
            Some(StreamInfo { target_device_id })
        })
        .collect())
}

fn list_sink_inputs_verbose() -> Result<Vec<(u32, String)>, String> {
    let text = pactl(&["list", "sink-inputs"])?;
    let mut inputs = Vec::new();
    let mut current_id = None;

    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Sink Input #") {
            current_id = rest.trim().parse::<u32>().ok();
        } else if let Some(rest) = trimmed.strip_prefix("application.name = ") {
            if let Some(id) = current_id {
                inputs.push((id, rest.trim_matches('"').to_string()));
            }
        }
    }
    Ok(inputs)
}

pub fn list_pulse_modules() -> Result<Vec<(u32, String, String)>, String> {
    let text = pactl(&["list", "modules", "short"])?;
    let mut modules = Vec::new();
    for line in text.lines() {
        let mut parts = line.splitn(3, '\t');
        let head = parts.next().unwrap_or_default().trim();
        if let (Ok(id), Some(name)) = (head.parse::<u32>(), parts.next()) {
            let args = parts.next().unwrap_or_default().trim().to_string();
            modules.push((id, name.trim().to_string(), args));
        }
    }
    Ok(modules)
}

pub fn unload_pulse_module(module_id: u32) -> Result<(), String> {
    pactl(&["unload-module", &module_id.to_string()]).map(|_| ())
}

fn has_module_arg(args: &str, key: &str, value: &str) -> bool {
    if value.is_empty() {
        return false;
    }
    let needle = format!("{key}={value}");
    args.split_whitespace().any(|token| token == needle)
}

fn pw_nodes() -> Vec<(u32, String)> {
    match Command::new("pw-cli").args(["ls", "Node"]).output() {
        Ok(out) => {
            let text = String::from_utf8_lossy(&out.stdout);
            let mut nodes = Vec::new();
            let mut current_id: Option<u32> = None;
            for line in text.lines() {
                let trimmed = line.trim();
                if let Some(rest) = trimmed.strip_prefix("id ") {
                    current_id = rest
                        .split(',')
                        .next()
                        .and_then(|id| id.trim().parse::<u32>().ok());
                } else if let Some(rest) = trimmed.strip_prefix("node.name = ") {
                    if let Some(id) = current_id {
                        nodes.push((id, rest.trim().trim_matches('"').to_string()));
                    }
                }
            }
            nodes
        }
        Err(_) => Vec::new(),
    }
}

pub fn pw_node_exists(name: &str) -> bool {
    !name.is_empty() && pw_nodes().iter().any(|(_, node_name)| node_name == name)
}

pub fn cleanup_pipewire_nodes(speaker_name: &str, mic_name: &str) -> usize {
    let mut destroyed = 0;
    for (id, name) in pw_nodes() {
        if name == speaker_name || name == mic_name {
            let _ = Command::new("pw-cli").args(["destroy", &id.to_string()]).output();
            destroyed += 1;
        }
    }
    destroyed
}

fn link_ports(out_port: &str, in_port: &str) -> Result<(), String> {
    let output = Command::new("pw-link")
        .args([out_port, in_port])
        .output()
        .map_err(|e| format!("Failed to run pw-link: {e}"))?;

    if output.status.success() || ports_are_linked(out_port, in_port) {
        return Ok(());
    }

    let detail = String::from_utf8_lossy(&output.stderr);
    Err(format!("{out_port} -> {in_port} failed: {}", detail.trim()))
}

fn unlink_ports(out_port: &str, in_port: &str) -> Result<(), String> {
    let _ = Command::new("pw-link")
        .args(["-d", out_port, in_port])
        .output();
    Ok(())
}

fn parse_links(text: &str) -> Vec<(String, String)> {
    let mut links = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if line.starts_with(char::is_whitespace) {
            let target = trimmed.split_whitespace().last().unwrap_or_default();
            if !target.is_empty() && !current.is_empty() {
                links.push((current.clone(), target.to_string()));
            }
        } else {
            current = trimmed.to_string();
        }
    }
    links
}

pub fn ports_are_linked(out_port: &str, in_port: &str) -> bool {
    match Command::new("pw-link").arg("-l").output() {
        Ok(out) => parse_links(&String::from_utf8_lossy(&out.stdout))
            .iter()
            .any(|(a, b)| (a == out_port && b == in_port) || (a == in_port && b == out_port)),
        Err(_) => false,
    }
}

fn find_hevoicechanger_output_ports() -> Vec<String> {
    let Ok(out) = Command::new("pw-link").arg("-o").output() else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .map(str::trim)
        .filter(|line| line.contains("hevoicechanger") && line.contains("output_"))
        .map(String::from)
        .collect()
}

fn get_connected_input_ports_for(out_ports: &[String]) -> Vec<String> {
    let Ok(out) = Command::new("pw-link").arg("-l").output() else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let links = parse_links(&text);
    let mut targets = Vec::new();

    for out_p in out_ports {
        for (src, dst) in &links {
            if src == out_p && !targets.contains(dst) {
                targets.push(dst.clone());
            }
        }
    }
    targets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtual_mic_config_defaults() {
        let config = VirtualMicConfig::default();
        assert_eq!(config.speaker_name, DEFAULT_SINK_NAME);
        assert_eq!(config.mic_name, DEFAULT_MIC_NAME);
        assert_eq!(config.channels, 2);
        assert!(VirtualMicConfig::is_valid_name(&config.mic_name));
        assert!(VirtualMicConfig::is_valid_name(&config.speaker_name));
    }

    #[test]
    fn test_valid_name_validation() {
        assert!(VirtualMicConfig::is_valid_name("HEVoiceMic"));
        assert!(VirtualMicConfig::is_valid_name("my_virtual_mic-1.0"));
        assert!(!VirtualMicConfig::is_valid_name(""));
        assert!(!VirtualMicConfig::is_valid_name("invalid name with spaces"));
        assert!(!VirtualMicConfig::is_valid_name("invalid/name"));
    }

    #[test]
    fn test_parse_links() {
        let sample = "alsa_output.pci:playback_FL\n  |-> alsa_input:capture_FL\nalsa_output.pci:playback_FR\n  |-> alsa_input:capture_FR\n";
        let links = parse_links(sample);
        assert_eq!(links.len(), 2);
        assert_eq!(links[0], ("alsa_output.pci:playback_FL".to_string(), "alsa_input:capture_FL".to_string()));
        assert_eq!(links[1], ("alsa_output.pci:playback_FR".to_string(), "alsa_input:capture_FR".to_string()));
    }
}
