from huggingface_hub import snapshot_download

# Download the tokenizer (required)
print("Downloading tokenizer...")
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-Tokenizer-12Hz",
    local_dir="./Qwen3-TTS-Tokenizer-12Hz",
    local_dir_use_symlinks=False
)

# Download a model variant (choose one based on your needs)
print("Downloading model...")
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",  # or 0.6B for smaller size
    local_dir="./Qwen3-TTS-12Hz-1.7B-CustomVoice",
    local_dir_use_symlinks=False
)

print("Download complete!")
