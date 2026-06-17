from huggingface_hub import snapshot_download

# Download the tokenizer (required)
print("Downloading tokenizer...")
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-Tokenizer-12Hz",
    local_dir="./Qwen3-TTS-Tokenizer-12Hz",
    local_dir_use_symlinks=False,
)

# VoiceDesign model — supports generate_voice_design() for natural-language voice creation
print("Downloading VoiceDesign model...")
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    local_dir="./Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    local_dir_use_symlinks=False,
)

# Base model — supports generate_voice_clone() for consistent voice across chunks
# Public, Apache 2.0, no token required
print("Downloading base model...")
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    local_dir="./Qwen3-TTS-12Hz-1.7B-Base",
    local_dir_use_symlinks=False,
)

print("\nDownload complete!")
