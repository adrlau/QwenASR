//! Cross-platform live audio capture via CPAL.
//!
//! Uses the native host backend on each platform
//! (CoreAudio on macOS, ALSA/PipeWire on Linux).

#![cfg(any(target_os = "linux", target_os = "macos"))]

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::mpsc;

/// An audio input device.
pub struct AudioDevice {
    pub id: String,
    pub name: String,
    pub input_channels: u32,
}

/// Capture handle — drop to stop capture.
pub struct CaptureHandle {
    _stream: cpal::Stream,
}

/// Get the list of audio input devices.
pub fn list_input_devices() -> Vec<AudioDevice> {
    let host = cpal::default_host();
    let mut devices = Vec::new();

    let input_devices = match host.input_devices() {
        Ok(d) => d,
        Err(_) => return devices,
    };

    for device in input_devices {
        let name = device
            .name()
            .unwrap_or_else(|_| "Unknown device".to_string());
        let input_channels = device
            .default_input_config()
            .map(|cfg| cfg.channels() as u32)
            .unwrap_or(0);
        devices.push(AudioDevice {
            id: name.clone(),
            name,
            input_channels,
        });
    }

    devices
}

/// Find an input device by name (case-insensitive substring match).
pub fn find_device_by_name(name: &str) -> Option<AudioDevice> {
    let name_lower = name.to_lowercase();
    list_input_devices()
        .into_iter()
        .find(|d| d.name.to_lowercase().contains(&name_lower))
}

/// Get the default input device id.
pub fn default_input_device() -> Option<String> {
    let host = cpal::default_host();
    host.default_input_device().and_then(|d| d.name().ok())
}

/// Print all input devices to stderr.
pub fn print_devices() {
    let devices = list_input_devices();
    if devices.is_empty() {
        eprintln!("No audio input devices found.");
        return;
    }

    let default_id = default_input_device();
    eprintln!("Audio input devices:\n");
    for d in &devices {
        let marker = if Some(&d.id) == default_id.as_ref() {
            " (default)"
        } else {
            ""
        };
        eprintln!("  {:30} {} ch{}", d.name, d.input_channels, marker);
    }
    eprintln!();
}

/// Start capturing audio from a device.
/// Returns mono f32 chunks at the device sample rate.
pub fn start_capture(
    device_id: String,
) -> Result<(mpsc::Receiver<Vec<f32>>, CaptureHandle, f64), String> {
    let host = cpal::default_host();
    let mut selected = None;
    let input_devices = host
        .input_devices()
        .map_err(|e| format!("Cannot enumerate input devices: {e}"))?;
    for device in input_devices {
        if let Ok(name) = device.name() {
            if name == device_id {
                selected = Some(device);
                break;
            }
        }
    }
    let device = selected.ok_or_else(|| format!("Input device not found: {device_id}"))?;

    let default_config = device
        .default_input_config()
        .map_err(|e| format!("Cannot get default input config: {e}"))?;
    let sample_rate = default_config.sample_rate().0 as f64;
    let channels = default_config.channels() as usize;
    let stream_config: cpal::StreamConfig = default_config.clone().into();
    let sample_format = default_config.sample_format();

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let err_fn = |err| eprintln!("Audio capture error: {err}");

    let stream = match sample_format {
        cpal::SampleFormat::F32 => build_stream_f32(&device, &stream_config, channels, tx, err_fn)?,
        cpal::SampleFormat::I16 => build_stream_i16(&device, &stream_config, channels, tx, err_fn)?,
        cpal::SampleFormat::U16 => build_stream_u16(&device, &stream_config, channels, tx, err_fn)?,
        _ => {
            return Err(format!(
                "Unsupported input sample format: {sample_format:?}"
            ))
        }
    };

    stream
        .play()
        .map_err(|e| format!("Cannot start input stream: {e}"))?;

    Ok((rx, CaptureHandle { _stream: stream }, sample_rate))
}

fn build_stream_f32(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    tx: mpsc::Sender<Vec<f32>>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream, String> {
    device
        .build_input_stream(
            config,
            move |data: &[f32], _| {
                let mono = interleaved_to_mono(data, channels, |s| s);
                let _ = tx.send(mono);
            },
            err_fn,
            None,
        )
        .map_err(|e| format!("Cannot build f32 input stream: {e}"))
}

fn build_stream_i16(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    tx: mpsc::Sender<Vec<f32>>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream, String> {
    device
        .build_input_stream(
            config,
            move |data: &[i16], _| {
                let mono = interleaved_to_mono(data, channels, |s| s as f32 / 32768.0);
                let _ = tx.send(mono);
            },
            err_fn,
            None,
        )
        .map_err(|e| format!("Cannot build i16 input stream: {e}"))
}

fn build_stream_u16(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    tx: mpsc::Sender<Vec<f32>>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream, String> {
    device
        .build_input_stream(
            config,
            move |data: &[u16], _| {
                let mono = interleaved_to_mono(data, channels, |s| {
                    (s as f32 / u16::MAX as f32) * 2.0 - 1.0
                });
                let _ = tx.send(mono);
            },
            err_fn,
            None,
        )
        .map_err(|e| format!("Cannot build u16 input stream: {e}"))
}

fn interleaved_to_mono<T>(data: &[T], channels: usize, to_f32: impl Fn(T) -> f32) -> Vec<f32>
where
    T: Copy,
{
    if channels <= 1 {
        return data.iter().copied().map(to_f32).collect();
    }
    let mut out = Vec::with_capacity(data.len() / channels);
    for frame in data.chunks_exact(channels) {
        let sum: f32 = frame.iter().copied().map(&to_f32).sum();
        out.push(sum / channels as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::interleaved_to_mono;

    #[test]
    fn interleaved_stereo_f32_is_mixed_to_mono() {
        let input = [1.0_f32, -1.0_f32, 0.5_f32, 0.5_f32];
        let mono = interleaved_to_mono(&input, 2, |s| s);
        assert_eq!(mono, vec![0.0, 0.5]);
    }

    #[test]
    fn mono_i16_is_scaled_to_f32() {
        let input = [i16::MIN, 0_i16, i16::MAX];
        let mono = interleaved_to_mono(&input, 1, |s| s as f32 / 32768.0);
        assert_eq!(mono[0], -1.0);
        assert_eq!(mono[1], 0.0);
        assert!(mono[2] < 1.0);
    }
}
