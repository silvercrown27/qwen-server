import os
import time
import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Model path: use local directory if downloaded, otherwise auto-download from Hub
MODEL_LOCAL = os.path.abspath("./Qwen3-TTS-12Hz-1.7B-CustomVoice")
MODEL_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
model_path = MODEL_LOCAL if os.path.isdir(MODEL_LOCAL) else MODEL_HF_ID

# Enable TF32 on Ampere+ GPUs (SM 8.x) — uses tensor cores for matmul, ~2x throughput
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    import flash_attn  # noqa: F401
    attn_impl = "flash_attention_2"
except Exception:
    attn_impl = "sdpa"

print(f"Loading model from: {model_path}  (attn={attn_impl})")
model = Qwen3TTSModel.from_pretrained(
    model_path,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation=attn_impl,
)

# Compile the inner nn.Module — Qwen3TTSModel is a wrapper, not nn.Module itself
print("Compiling model with torch.compile...")
model.model = torch.compile(model.model, mode="reduce-overhead")
print("Model ready.")

# ---------------------------------------------------------------------------
# LESSON 1 — Japanese terms (English narration with Japanese words embedded)
# Speaker: ryan (clear English voice for language instruction)
# ---------------------------------------------------------------------------

japanese_segments = [
    # Greetings
    "Welcome to your Japanese lesson. Let's start with greetings. "
    "The most common way to say hello is こんにちは — konnichiwa. "
    "In the morning, you say おはようございます — ohayou gozaimasu, meaning good morning. "
    "And in the evening, こんばんは — konbanwa — means good evening.",

    # Thank you / sorry
    "Next, two of the most essential phrases. "
    "To say thank you, use ありがとうございます — arigatou gozaimasu. "
    "For a more casual thank you between friends, simply ありがとう — arigatou. "
    "To apologize or say excuse me, you say すみません — sumimasen. "
    "Remember, すみません is also used to get someone's attention, like saying 'excuse me' to a waiter.",

    # Numbers
    "Let's learn the numbers one through five. "
    "One is 一 — ichi. Two is 二 — ni. Three is 三 — san. "
    "Four is 四 — shi, or sometimes よん — yon. And five is 五 — go. "
    "You might recognize these from martial arts — いち、に、さん、し、ご — "
    "ichi, ni, san, shi, go. Great work!",
]

japanese_languages = ["english", "english", "english"]

# ---------------------------------------------------------------------------
# LESSON 2 — Chinese (Mandarin) terms (English narration with Chinese embedded)
# Speaker: vivian
# ---------------------------------------------------------------------------

chinese_segments = [
    # Greetings
    "Now let's explore some Mandarin Chinese. "
    "The universal greeting is 你好 — nǐ hǎo, literally meaning 'you good'. "
    "For a more formal hello, say 您好 — nín hǎo, where 您 is the respectful form of 'you'. "
    "To ask how someone is doing, say 你好吗 — nǐ hǎo ma? The word 吗 — ma — turns any statement into a yes-or-no question.",

    # Food
    "Food is a great way to connect with a language. "
    "The word for rice is 米饭 — mǐ fàn. Noodles are 面条 — miàn tiáo. "
    "To say 'I want to eat', say 我想吃 — wǒ xiǎng chī. "
    "So '我想吃面条' — wǒ xiǎng chī miàn tiáo — means 'I want to eat noodles'. "
    "And if something is delicious, say 好吃 — hǎo chī — literally 'good to eat'.",

    # Tones reminder
    "Mandarin has four tones, and they completely change meaning. "
    "Listen to the word 'ma' with different tones: "
    "妈 — mā — first tone, means mother. "
    "麻 — má — second tone, means hemp or numb. "
    "马 — mǎ — third tone, means horse. "
    "骂 — mà — fourth tone, means to scold. "
    "So 妈妈骂马吗 — māma mà mǎ ma — means 'does mother scold the horse?' — a classic tongue twister for tone practice!",
]

chinese_languages = ["english", "english", "english"]

# ---------------------------------------------------------------------------
# Generate Japanese lesson
# ---------------------------------------------------------------------------

print(f"\n--- Generating Japanese lesson ({len(japanese_segments)} segments, speaker: ryan) ---")
start = time.time()

wavs_jp, sr = model.generate_custom_voice(
    text=japanese_segments,
    language=japanese_languages,
    speaker=["ryan"] * len(japanese_segments),
    instruct=(
        "Warm, encouraging teacher voice. Speak English naturally at a steady pace. "
        "When saying Japanese words, slow down slightly and pronounce them with care, "
        "then return to a friendly, conversational tone for the English explanation."
    ),
)

elapsed = time.time() - start
print(f"Japanese lesson generated in {elapsed:.2f}s  (sample_rate={sr})")

jp_segments = []
for i, wav in enumerate(wavs_jp, start=1):
    wav = wav.astype(np.float32)
    seg_path = f"jp_segment_{i}.wav"
    sf.write(seg_path, wav, sr)
    print(f"  jp_segment_{i}.wav — {len(wav) / sr:.1f}s")
    jp_segments.append(wav)

# ---------------------------------------------------------------------------
# Generate Chinese lesson
# ---------------------------------------------------------------------------

print(f"\n--- Generating Chinese lesson ({len(chinese_segments)} segments, speaker: vivian) ---")
start = time.time()

wavs_zh, sr = model.generate_custom_voice(
    text=chinese_segments,
    language=chinese_languages,
    speaker=["vivian"] * len(chinese_segments),
    instruct=(
        "Warm, enthusiastic language teacher. Speak English clearly and conversationally. "
        "When pronouncing Chinese words, enunciate each tone with care and a brief natural pause "
        "before continuing the English explanation — as if helping a student absorb each new word."
    ),
)

elapsed = time.time() - start
print(f"Chinese lesson generated in {elapsed:.2f}s  (sample_rate={sr})")

zh_segments = []
for i, wav in enumerate(wavs_zh, start=1):
    wav = wav.astype(np.float32)
    seg_path = f"zh_segment_{i}.wav"
    sf.write(seg_path, wav, sr)
    print(f"  zh_segment_{i}.wav — {len(wav) / sr:.1f}s")
    zh_segments.append(wav)

# ---------------------------------------------------------------------------
# Combine: Japanese lesson → pause → Chinese lesson
# ---------------------------------------------------------------------------

silence_short = np.zeros(int(sr * 0.8), dtype=np.float32)   # 0.8s between segments
silence_long  = np.zeros(int(sr * 2.0), dtype=np.float32)   # 2.0s between lessons

jp_combined = np.concatenate([seg for pair in zip(jp_segments, [silence_short] * len(jp_segments)) for seg in pair])
zh_combined = np.concatenate([seg for pair in zip(zh_segments, [silence_short] * len(zh_segments)) for seg in pair])

combined = np.concatenate([jp_combined, silence_long, zh_combined])

out_path = "combined_output.wav"
sf.write(out_path, combined, sr)

total_duration = len(combined) / sr
file_size_kb = os.path.getsize(out_path) / 1024

print(f"\nSaved: {out_path}")
print(f"  Total duration : {total_duration:.1f}s")
print(f"  File size      : {file_size_kb:.1f} KB")
print(f"\nIndividual files:")
print(f"  jp_segment_1.wav — jp_segment_{len(jp_segments)}.wav  (Japanese lesson)")
print(f"  zh_segment_1.wav — zh_segment_{len(zh_segments)}.wav  (Chinese lesson)")
