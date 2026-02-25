import os
import time
import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

SPEAKER = "sohee"

# Model path: use local directory if downloaded, otherwise auto-download from Hub
MODEL_LOCAL = os.path.abspath("./Qwen3-TTS-12Hz-1.7B-CustomVoice")
MODEL_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
model_path = MODEL_LOCAL if os.path.isdir(MODEL_LOCAL) else MODEL_HF_ID

print(f"Loading model from: {model_path}")
model = Qwen3TTSModel.from_pretrained(
    model_path,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# --- Three text segments ---

texts = [
    "Imagine multiple base stations coordinating their beams toward a single fast-moving device "
    "at millimeter-wave frequencies. This is the core challenge that deep learning-based "
    "coordinated beamforming is designed to solve.",

    "The paper tackles a fundamental problem in fifth generation networks. Millimeter-wave base "
    "stations equipped with massive antenna arrays must rapidly find and maintain precise signal "
    "beams for users moving at high speed. Traditional scanning approaches waste too much time "
    "and bandwidth on training overhead.",

    "The paper proposes three deep learning solutions for coordinated beamforming. Each strategy "
    "trains a neural network to predict the best beam combination across multiple base stations, "
    "dramatically reducing the pilot overhead compared to traditional exhaustive beam scanning methods.",
]

print(f"\nGenerating {len(texts)} segments for speaker '{SPEAKER}' (upbeat/enticing)...")
start = time.time()

# generate_custom_voice handles generation + decoding in one call
# instruct sets the emotional style
wavs, sr = model.generate_custom_voice(
    text=texts,
    language=["english"] * len(texts),
    speaker=[SPEAKER] * len(texts),
    instruct=[
        "Speak with a sense of wonder and discovery, building intrigue — slow-paced with a compelling, cinematic pitch",
        "Authoritative and serious, like a documentary narrator presenting a critical problem — measured pace with dramatic gravitas",
        "Confident and forward-looking, rising energy toward the conclusion — warm storyteller tone with a sense of resolution",
    ],
)

elapsed = time.time() - start
print(f"Generation complete in {elapsed:.2f}s  (sample_rate={sr})")

# --- Save individual segments ---

segments = []
for i, wav in enumerate(wavs, start=1):
    wav = wav.astype(np.float32)
    seg_path = f"segment_{i}.wav"
    sf.write(seg_path, wav, sr)
    print(f"  segment_{i}.wav — {len(wav) / sr:.1f}s")
    segments.append(wav)

# --- Combine with 1-second silence between segments ---

silence = np.zeros(sr, dtype=np.float32)

combined = np.concatenate([
    segments[0], silence,
    segments[1], silence,
    segments[2],
])

out_path = "combined_output.wav"
sf.write(out_path, combined, sr)

total_duration = len(combined) / sr
file_size_kb = os.path.getsize(out_path) / 1024

print(f"\nSaved: {out_path}")
print(f"  Total duration : {total_duration:.1f}s")
print(f"  File size      : {file_size_kb:.1f} KB")
