//! Model download from HuggingFace with progress display.

use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const SAFETENSORS_EXT: &str = ".safetensors";
const SAFETENSORS_INDEX_EXT: &str = ".safetensors.index.json";

// ========================================================================
// Model Registry
// ========================================================================

pub struct ModelInfo {
    pub name: &'static str,
    pub repo: &'static str,
    pub files: &'static [&'static str],
    pub description: &'static str,
}

pub const KNOWN_MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "qwen3-asr-0.6b",
        repo: "Qwen/Qwen3-ASR-0.6B",
        files: &["model.safetensors", "vocab.json", "merges.txt"],
        description: "Qwen3-ASR 0.6B — fast, ~490 MB",
    },
    ModelInfo {
        name: "qwen3-asr-1.7b",
        repo: "Qwen/Qwen3-ASR-1.7B",
        files: &[
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "vocab.json",
            "merges.txt",
        ],
        description: "Qwen3-ASR 1.7B — higher accuracy, ~3.4 GB",
    },
    ModelInfo {
        name: "qwen3-aligner-0.6b",
        repo: "Qwen/Qwen3-ASR-ForcedAligner-0.6B",
        files: &[
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "vocab.json",
            "merges.txt",
        ],
        description: "Qwen3-ASR ForcedAligner 0.6B — word-level timestamps, ~1.6 GB",
    },
];

pub fn find_model(name: &str) -> Option<&'static ModelInfo> {
    let name_lower = name.to_lowercase();
    KNOWN_MODELS.iter().find(|m| m.name == name_lower)
}

fn home_dir() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("HOME") {
        return Some(PathBuf::from(home));
    }
    std::env::var_os("USERPROFILE").map(PathBuf::from)
}

pub fn default_models_root() -> PathBuf {
    let root = match env::var("XDG_DATA_HOME") {
        Ok(val) => Path::new(&val).join("qwen-asr").join("models"),
        Err(_) => {
            #[cfg(target_os = "linux")]
            {
                if let Some(home) = home_dir() {
                    home.join(".local")
                        .join("share")
                        .join("qwen-asr")
                        .join("models")
                } else {
                    PathBuf::from(".").join(".qwen-asr").join("models")
                }
            }

            #[cfg(target_os = "macos")]
            {
                if let Some(home) = home_dir() {
                    home.join("Library")
                        .join("Application Support")
                        .join("qwen-asr")
                        .join("models")
                } else {
                    PathBuf::from(".").join(".qwen-asr").join("models")
                }
            }

            #[cfg(target_os = "windows")]
            {
                if let Some(appdata) = std::env::var_os("APPDATA") {
                    PathBuf::from(appdata).join("qwen-asr").join("models")
                } else if let Some(home) = home_dir() {
                    home.join("AppData")
                        .join("Roaming")
                        .join("qwen-asr")
                        .join("models")
                } else {
                    PathBuf::from(".").join(".qwen-asr").join("models")
                }
            }

            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            {
                home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".qwen-asr")
                    .join("models")
            }
        }
    };

    // Ensure the directory exists so callers can use it immediately.
    if let Err(e) = fs::create_dir_all(&root) {
        eprintln!(
            "Warning: could not create models directory {}: {}",
            root.display(),
            e
        );
    }
    root
}

fn sanitize_model_dir_name(name: &str) -> String {
    let trimmed = name.trim().trim_end_matches('/');
    let cleaned: String = trimmed
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '/' {
                c
            } else {
                '-'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "model".to_string()
    } else {
        cleaned
    }
}

pub fn resolve_model_dir(model_input: &str) -> PathBuf {
    if looks_like_path(model_input) {
        PathBuf::from(model_input)
    } else {
        default_models_root().join(sanitize_model_dir_name(model_input))
    }
}

fn looks_like_path(value: &str) -> bool {
    let p = Path::new(value);
    p.is_absolute()
        || value.starts_with('.')
        || value.starts_with('~')
        || (cfg!(windows) && value.contains(':'))
}

pub fn is_hf_repo_id(value: &str) -> bool {
    if value.starts_with("/")
        || value.starts_with("./")
        || value.starts_with("../")
        || value.starts_with("~")
        || value.contains("\\")
    {
        return false;
    }
    let mut parts = value.split('/');
    let owner = parts.next().unwrap_or("");
    let name = parts.next().unwrap_or("");
    parts.next().is_none() && !owner.is_empty() && !name.is_empty()
}

// ========================================================================
// List Models
// ========================================================================

pub fn list_models() {
    eprintln!("Available models:\n");
    for m in KNOWN_MODELS {
        eprintln!("  {:<24} {}", m.name, m.description);
    }
    eprintln!();
    eprintln!("Usage: qwen-asr download <model-name|hf-repo> [--output <dir>]");
    eprintln!("Default output root: {}", default_models_root().display());
}

// ========================================================================
// Download
// ========================================================================

fn hf_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{}/resolve/main/{}", repo, file)
}

fn hf_model_api_url(repo: &str) -> String {
    format!("https://huggingface.co/api/models/{}", repo)
}

fn list_hf_repo_files(repo: &str) -> Result<Vec<String>, String> {
    if !is_hf_repo_id(repo) {
        return Err(format!(
            "Invalid Hugging Face repo ID '{}'. Expected format: owner/repo-name",
            repo
        ));
    }
    let resp = ureq::get(&hf_model_api_url(repo)).call().map_err(|e| {
        format!(
            "Failed to fetch Hugging Face repo metadata for '{}': {}",
            repo, e
        )
    })?;
    let body = resp
        .into_string()
        .map_err(|e| format!("Failed to read Hugging Face response: {}", e))?;

    let mut files = Vec::new();
    for part in body.split("\"rfilename\":\"").skip(1) {
        if let Some((raw, _)) = part.split_once('"') {
            let name = raw.replace("\\/", "/");
            if !name.is_empty() {
                files.push(name);
            }
        }
    }
    files.sort();
    files.dedup();

    if files.is_empty() {
        Err(format!(
            "No files found for Hugging Face repo '{}'. Check that the repo exists and is public.",
            repo
        ))
    } else {
        Ok(files)
    }
}

fn select_repo_model_files(repo_files: &[String]) -> Vec<String> {
    const COMMON_EXTRA_FILES: &[&str] = &[
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer.model",
        "preprocessor_config.json",
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
    ];

    let mut wanted: Vec<String> = repo_files
        .iter()
        .filter(|name| name.ends_with(SAFETENSORS_EXT) || name.ends_with(SAFETENSORS_INDEX_EXT))
        .cloned()
        .collect();

    for name in COMMON_EXTRA_FILES {
        if repo_files.iter().any(|n| n == name) {
            wanted.push((*name).to_string());
        }
    }

    wanted.sort();
    wanted.dedup();
    wanted
}

/// Format bytes as human-readable size.
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Download a single file with progress display and resume support.
fn download_file(url: &str, dest: &Path) -> Result<(), String> {
    let mut start_byte: u64 = 0;

    // Check for partial download (resume support)
    let part_path = dest.with_extension(
        dest.extension()
            .map(|e| format!("{}.part", e.to_string_lossy()))
            .unwrap_or_else(|| "part".to_string()),
    );
    if part_path.exists() {
        start_byte = fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
    }

    // Already fully downloaded?
    if dest.exists() {
        return Ok(());
    }

    // Build request
    let mut req = ureq::get(url);
    if start_byte > 0 {
        req = req.set("Range", &format!("bytes={}-", start_byte));
        eprint!("  Resuming from {} ... ", format_bytes(start_byte));
    }

    let resp = req
        .call()
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    // Parse content length
    let total_bytes = if start_byte > 0 {
        // For Range requests, Content-Range: bytes start-end/total
        resp.header("Content-Range")
            .and_then(|cr| cr.rsplit('/').next())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    } else {
        resp.header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    };

    let mut reader = resp.into_reader();
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&part_path)
        .map_err(|e| format!("Cannot open {}: {}", part_path.display(), e))?;

    let mut downloaded = start_byte;
    let mut buf = vec![0u8; 256 * 1024]; // 256 KB buffer
    let mut last_progress = std::time::Instant::now();
    let start_time = std::time::Instant::now();

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("Read error: {}", e))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("Write error: {}", e))?;
        downloaded += n as u64;

        // Update progress ~4 times per second
        let now = std::time::Instant::now();
        if now.duration_since(last_progress).as_millis() >= 250 || n == 0 {
            last_progress = now;
            let elapsed = now.duration_since(start_time).as_secs_f64();
            let speed = if elapsed > 0.0 {
                (downloaded - start_byte) as f64 / elapsed
            } else {
                0.0
            };

            if total_bytes > 0 {
                let pct = (downloaded as f64 / total_bytes as f64 * 100.0).min(100.0);
                eprint!(
                    "\r  {} / {} ({:.0}%) {}/s    ",
                    format_bytes(downloaded),
                    format_bytes(total_bytes),
                    pct,
                    format_bytes(speed as u64),
                );
            } else {
                eprint!(
                    "\r  {} downloaded, {}/s    ",
                    format_bytes(downloaded),
                    format_bytes(speed as u64),
                );
            }
        }
    }

    eprintln!(); // newline after progress

    // Rename .part to final destination
    fs::rename(&part_path, dest).map_err(|e| {
        format!(
            "Cannot rename {} → {}: {}",
            part_path.display(),
            dest.display(),
            e
        )
    })?;

    Ok(())
}

/// Download all files for a model.
pub fn download_model(model: &ModelInfo, output_dir: &str) -> Result<(), String> {
    let dir = PathBuf::from(output_dir);
    fs::create_dir_all(&dir)
        .map_err(|e| format!("Cannot create directory {}: {}", output_dir, e))?;

    let total_files = model.files.len();
    for (i, file_name) in model.files.iter().enumerate() {
        let dest = dir.join(file_name);
        if dest.exists() {
            eprintln!(
                "[{}/{}] {} — already exists, skipping",
                i + 1,
                total_files,
                file_name
            );
            continue;
        }

        let url = hf_url(model.repo, file_name);
        eprintln!("[{}/{}] Downloading {} ...", i + 1, total_files, file_name);
        download_file(&url, &dest)?;
    }

    eprintln!("\n✓ Model '{}' downloaded to {}", model.name, output_dir);
    Ok(())
}

pub fn download_repo_model(repo: &str, output_dir: &str) -> Result<(), String> {
    let dir = PathBuf::from(output_dir);
    fs::create_dir_all(&dir)
        .map_err(|e| format!("Cannot create directory {}: {}", output_dir, e))?;

    let repo_files = list_hf_repo_files(repo)?;
    let files = select_repo_model_files(&repo_files);
    if files.is_empty() {
        return Err(format!(
            "No model files found in '{}'. Expected safetensors and tokenizer files.",
            repo
        ));
    }

    let total_files = files.len();
    for (i, file_name) in files.iter().enumerate() {
        let dest = dir.join(file_name);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Cannot create directory {}: {}", parent.display(), e))?;
        }
        if dest.exists() {
            eprintln!(
                "[{}/{}] {} — already exists, skipping",
                i + 1,
                total_files,
                file_name
            );
            continue;
        }

        let url = hf_url(repo, file_name);
        eprintln!("[{}/{}] Downloading {} ...", i + 1, total_files, file_name);
        download_file(&url, &dest)?;
    }

    eprintln!("\n✓ Model '{}' downloaded to {}", repo, output_dir);
    Ok(())
}

// ========================================================================
// Interactive Prompt
// ========================================================================

/// Prompt user to download a model. Returns true if they accepted.
pub fn prompt_download(model_name: &str) -> bool {
    let model = match find_model(model_name) {
        Some(m) => m,
        None => return false,
    };

    eprintln!("Model directory '{}' not found.\n", model_name);
    eprintln!("  {} — {}\n", model.name, model.description);
    eprint!("Download now? [Y/n]: ");
    io::stderr().flush().ok();

    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        return false;
    }
    let answer = input.trim().to_lowercase();
    answer.is_empty() || answer == "y" || answer == "yes"
}

// ========================================================================
// CLI Entry Point
// ========================================================================

/// Handle the `download` subcommand. Returns true if handled (caller should exit).
pub fn handle_download_command(args: &[String]) -> bool {
    // Parse: download [--list] [<model-name>] [--output <dir>]
    let mut model_name: Option<String> = None;
    let mut output_dir: Option<String> = None;
    let mut show_list = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--list" | "-l" => {
                show_list = true;
            }
            "--output" | "-o" => {
                i += 1;
                output_dir = args.get(i).cloned();
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: qwen-asr download [--list] [<model-name|hf-repo>] [--output <dir>]\n"
                );
                eprintln!("Options:");
                eprintln!("  --list, -l       List available models");
                eprintln!(
                    "  --output, -o     Download directory (default: {}/<model-name>/)",
                    default_models_root().display()
                );
                eprintln!("  -h, --help       Show this help");
                return true;
            }

            other => {
                if other.starts_with('-') {
                    eprintln!("Unknown option for download: {}", other);
                    return true;
                }
                model_name = Some(other.to_string());
            }
        }
        i += 1;
    }

    if show_list || model_name.is_none() {
        list_models();
        return true;
    }

    let name = model_name.unwrap();
    if looks_like_path(&name) {
        eprintln!(
            "Error: '{}' appears to be a local path. The download command expects a model name or Hugging Face repository ID.\n",
            name
        );
        eprintln!("To download a model to a specific directory, use:\n  qwen-asr download <model-name> --output /your/path/\n");
        return true;
    }

    if let Some(model) = find_model(&name) {
        let path = resolve_model_dir(model.name);
        let dir = output_dir.unwrap_or_else(|| path.to_string_lossy().to_string());
        match download_model(model, &dir) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("\nError: {}", e);
                std::process::exit(1);
            }
        }
    } else if is_hf_repo_id(&name) {
        let path = resolve_model_dir(&name);
        let dir = output_dir.unwrap_or_else(|| path.to_string_lossy().to_string());
        match download_repo_model(&name, &dir) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("\nError: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!(
            "Invalid model name or Hugging Face repo ID format: '{}'\n",
            name
        );
        list_models();
        std::process::exit(1);
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_named_model_uses_default_root() {
        let dir = resolve_model_dir("qwen3-asr-0.6b");
        assert!(dir.ends_with("qwen3-asr-0.6b"));
    }

    #[test]
    fn repo_id_detection() {
        assert!(is_hf_repo_id("mlx-community/Qwen3-ASR-0.6B-4bit"));
        assert!(!is_hf_repo_id("./models/qwen"));
        assert!(!is_hf_repo_id("/tmp/qwen"));
        assert!(!is_hf_repo_id("qwen3-asr-0.6b"));
    }

    #[test]
    fn repo_file_selection_prefers_model_and_tokenizer() {
        let files = vec![
            "README.md".to_string(),
            "model.safetensors".to_string(),
            "tokenizer.json".to_string(),
            "config.json".to_string(),
        ];
        let selected = select_repo_model_files(&files);
        assert_eq!(
            selected,
            vec![
                "config.json".to_string(),
                "model.safetensors".to_string(),
                "tokenizer.json".to_string()
            ]
        );
    }
}
