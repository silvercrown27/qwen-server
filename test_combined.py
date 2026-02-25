import os
import time
import numpy as np
import soundfile as sf
from vllm import LLM, SamplingParams
from qwen_tts import QwenTTSProcessor

SAMPLE_RATE = 24000
SPEAKER = "sohee"

# Model path: use local directory if downloaded, otherwise auto-download from Hub
MODEL_LOCAL = os.path.abspath("./Qwen3-TTS-12Hz-1.7B-CustomVoice")
MODEL_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
model_path = MODEL_LOCAL if os.path.isdir(MODEL_LOCAL) else MODEL_HF_ID

TOKENIZER_LOCAL = os.path.abspath("./Qwen3-TTS-Tokenizer-12Hz")
TOKENIZER_HF_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
tokenizer_path = TOKENIZER_LOCAL if os.path.isdir(TOKENIZER_LOCAL) else TOKENIZER_HF_ID

print(f"Loading model from: {model_path}")
llm = LLM(
    model=model_path,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
)

print(f"Loading tokenizer/processor from: {tokenizer_path}")
processor = QwenTTSProcessor.from_pretrained(tokenizer_path)

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

# Build prompts — <|emotion|>happy drives the upbeat/enticing delivery
prompts = [
    f"<|speaker|>{SPEAKER}<|language|>english<|emotion|>happy<|text|>{text}<|audio|>"
    for text in texts
]

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=4096,
)

print(f"\nGenerating {len(texts)} segments for speaker '{SPEAKER}' (upbeat/happy emotion)...")
start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start
print(f"Generation complete in {elapsed:.2f}s")

# --- Decode audio tokens and save individual segments ---

segments = []
for i, output in enumerate(outputs, start=1):
    token_ids = output.outputs[0].token_ids
    waveform = processor.decode(token_ids)
    waveform = waveform.astype(np.float32)

    seg_path = f"segment_{i}.wav"
    sf.write(seg_path, waveform, SAMPLE_RATE)
    duration = len(waveform) / SAMPLE_RATE
    print(f"  segment_{i}.wav — {duration:.1f}s  ({len(token_ids)} tokens)")
    segments.append(waveform)

# --- Combine with 1-second silence between segments ---

silence = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second at 24 kHz

combined = np.concatenate([
    segments[0], silence,
    segments[1], silence,
    segments[2],
])

out_path = "combined_output.wav"
sf.write(out_path, combined, SAMPLE_RATE)

total_duration = len(combined) / SAMPLE_RATE
file_size_kb = os.path.getsize(out_path) / 1024

print(f"\nSaved: {out_path}")
print(f"  Total duration : {total_duration:.1f}s")
print(f"  File size      : {file_size_kb:.1f} KB")
