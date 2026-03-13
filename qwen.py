import os
from huggingface_hub import snapshot_download

# The base model (Qwen3-TTS-12Hz-1.7B) is gated on HuggingFace.
# Before running this script:
#   1. Accept the model terms at: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B
#   2. Generate an access token at: https://huggingface.co/settings/tokens
#   3. Run:  HF_TOKEN=hf_xxx python qwen.py
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set — base model download will fail if gated.")
    print("  export HF_TOKEN=hf_xxxxxxxxxxxxxxxx")

# Download the tokenizer (required)
print("\nDownloading tokenizer...")
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-Tokenizer-12Hz",
    local_dir="./Qwen3-TTS-Tokenizer-12Hz",
    local_dir_use_symlinks=False,
    token=HF_TOKEN,
)

# CustomVoice model — preset speakers (sohee, ryan, vivian, etc.)
print("Downloading CustomVoice model...")
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    local_dir="./Qwen3-TTS-12Hz-1.7B-CustomVoice",
    local_dir_use_symlinks=False,
    token=HF_TOKEN,
)

# Base model — supports generate_voice_clone() for consistent voice across chunks
print("Downloading base model (requires HF_TOKEN + accepted terms)...")
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-12Hz-1.7B",
    local_dir="./Qwen3-TTS-12Hz-1.7B",
    local_dir_use_symlinks=False,
    token=HF_TOKEN,
)

print("\nDownload complete!")
