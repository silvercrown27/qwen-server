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

print("Compiling model with torch.compile...")
model.model = torch.compile(model.model, mode="reduce-overhead")
print("Model ready.")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def rms_normalize(wav: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    """Scale wav so its RMS matches target_rms.  Prevents chunk-to-chunk loudness jumps."""
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < 1e-9:
        return wav
    return wav * (target_rms / rms)


def crossfade_join(chunks: list[np.ndarray], sr: int,
                   gap_ms: int = 60, fade_ms: int = 10) -> np.ndarray:
    """
    Concatenate audio chunks with a short silence gap and linear cross-fade at each
    boundary to avoid clicks.  All chunks are RMS-normalised before joining.

    gap_ms   — silence inserted between every chunk (milliseconds)
    fade_ms  — fade-out / fade-in applied at the boundary (milliseconds)
    """
    target_rms = float(np.mean([np.sqrt(np.mean(c ** 2)) for c in chunks if len(c) > 0]))
    target_rms = max(target_rms, 0.02)

    normalised = [rms_normalize(c.astype(np.float32), target_rms) for c in chunks]

    gap    = np.zeros(int(sr * gap_ms  / 1000), dtype=np.float32)
    fade_n = int(sr * fade_ms / 1000)

    fade_out = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)

    parts = []
    for i, chunk in enumerate(normalised):
        c = chunk.copy()
        # Fade the tail of this chunk
        if len(c) >= fade_n:
            c[-fade_n:] *= fade_out
        # Fade the head of this chunk
        if len(c) >= fade_n:
            c[:fade_n] *= fade_in
        parts.append(c)
        if i < len(normalised) - 1:
            parts.append(gap)

    return np.concatenate(parts)


def generate_lesson(chunks: list[tuple[str, str]], speaker: str,
                    instruct: str) -> tuple[np.ndarray, int]:
    """
    Generate a language lesson by splitting at language boundaries.

    chunks  : list of (text, language) tuples
    speaker : single speaker name applied to every chunk
    instruct: ONE shared instruction string — critical for voice consistency.
              Do NOT vary this per-language; consistent instruction = consistent prosody.

    Returns (audio_array, sample_rate).
    """
    texts = [t for t, _ in chunks]
    langs = [l for _, l in chunks]

    # Single instruct string applied to all segments — keeps voice consistent
    wavs, sr = model.generate_custom_voice(
        text=texts,
        language=langs,
        speaker=[speaker] * len(texts),
        instruct=instruct,        # str, not list — uniform voice profile
    )

    audio = crossfade_join([w for w in wavs], sr, gap_ms=70, fade_ms=12)
    return audio, sr


# ---------------------------------------------------------------------------
# LESSON 1 — Japanese
# Single instruct: teacher voice that adapts pronunciation per language tag
# ---------------------------------------------------------------------------

JP_INSTRUCT = (
    "Warm, encouraging language teacher with a clear and steady voice. "
    "Speak at a measured pace throughout — both English and Japanese words should "
    "feel like they come from the same calm, friendly speaker."
)

jp_chunks = [
    # Greetings
    ("Welcome to your Japanese lesson. Let's start with greetings. "
     "The most common way to say hello is",                              "english"),
    ("こんにちは",                                                        "japanese"),
    ("In the morning you say",                                            "english"),
    ("おはようございます",                                                "japanese"),
    ("meaning good morning. And in the evening,",                         "english"),
    ("こんばんは",                                                        "japanese"),
    ("means good evening.",                                               "english"),

    # Thank you / excuse me
    ("Now let's learn some essential phrases. To say thank you use",      "english"),
    ("ありがとうございます",                                              "japanese"),
    ("For a casual thank you between friends, simply",                    "english"),
    ("ありがとう",                                                        "japanese"),
    ("To apologize or get someone's attention say",                       "english"),
    ("すみません",                                                        "japanese"),
    ("This works like excuse me or I'm sorry depending on context.",      "english"),

    # Numbers 1–5
    ("Now let's count from one to five. One is",                          "english"),
    ("いち",                                                              "japanese"),
    ("Two is",                                                            "english"),
    ("に",                                                                "japanese"),
    ("Three is",                                                          "english"),
    ("さん",                                                              "japanese"),
    ("Four is",                                                           "english"),
    ("し、またはよん",                                                    "japanese"),
    ("And five is",                                                       "english"),
    ("ご",                                                                "japanese"),
    ("You might recognise those from martial arts — great work!",         "english"),
]

# ---------------------------------------------------------------------------
# LESSON 2 — Mandarin Chinese
# ---------------------------------------------------------------------------

ZH_INSTRUCT = (
    "Warm, enthusiastic language teacher with a clear and steady voice. "
    "Speak at a measured pace throughout — both English and Chinese words should "
    "feel like they come from the same calm, friendly speaker."
)

zh_chunks = [
    # Greetings
    ("Now let's explore some Mandarin Chinese. "
     "The most common greeting is",                                        "english"),
    ("你好",                                                               "chinese"),
    ("which literally means you good. For a more formal hello say",        "english"),
    ("您好",                                                               "chinese"),
    ("where 您 is the respectful form of you. "
     "To ask how someone is doing say",                                    "english"),
    ("你好吗",                                                             "chinese"),
    ("The word",                                                           "english"),
    ("吗",                                                                 "chinese"),
    ("turns any statement into a yes or no question.",                     "english"),

    # Food vocab
    ("Food is a wonderful way to connect with a language. "
     "The word for rice is",                                               "english"),
    ("米饭",                                                               "chinese"),
    ("Noodles are",                                                        "english"),
    ("面条",                                                               "chinese"),
    ("To say I want to eat say",                                           "english"),
    ("我想吃",                                                             "chinese"),
    ("So I want to eat noodles is",                                        "english"),
    ("我想吃面条",                                                         "chinese"),
    ("And if something is delicious say",                                  "english"),
    ("好吃",                                                               "chinese"),
    ("which literally means good to eat.",                                 "english"),

    # The four tones
    ("Mandarin has four tones that completely change meaning. "
     "Listen to the syllable ma in each tone. First tone, meaning mother:", "english"),
    ("妈",                                                                 "chinese"),
    ("Second tone, meaning hemp or numb:",                                  "english"),
    ("麻",                                                                 "chinese"),
    ("Third tone, meaning horse:",                                          "english"),
    ("马",                                                                 "chinese"),
    ("Fourth tone, meaning to scold:",                                      "english"),
    ("骂",                                                                 "chinese"),
    ("Now here is a classic tongue twister using all four:",                "english"),
    ("妈妈骂马吗",                                                         "chinese"),
    ("Which means: does mother scold the horse? Fantastic listening!",      "english"),
]

# ---------------------------------------------------------------------------
# Generate both lessons
# ---------------------------------------------------------------------------

print("\n--- Generating Japanese lesson ---")
t0 = time.time()
jp_audio, sr = generate_lesson(jp_chunks, speaker="ryan", instruct=JP_INSTRUCT)
print(f"Japanese lesson: {len(jp_audio)/sr:.1f}s  ({time.time()-t0:.1f}s gen time)")
sf.write("japanese_lesson.wav", jp_audio, sr)

print("\n--- Generating Chinese lesson ---")
t0 = time.time()
zh_audio, sr = generate_lesson(zh_chunks, speaker="vivian", instruct=ZH_INSTRUCT)
print(f"Chinese lesson:  {len(zh_audio)/sr:.1f}s  ({time.time()-t0:.1f}s gen time)")
sf.write("chinese_lesson.wav", zh_audio, sr)

# ---------------------------------------------------------------------------
# Combine: Japanese → 3 s pause → Chinese
# ---------------------------------------------------------------------------

pause = np.zeros(int(sr * 3.0), dtype=np.float32)
combined = np.concatenate([jp_audio, pause, zh_audio])

out_path = "combined_output.wav"
sf.write(out_path, combined, sr)

total_duration = len(combined) / sr
file_size_kb = os.path.getsize(out_path) / 1024

print(f"\nSaved:")
print(f"  japanese_lesson.wav  — {len(jp_audio)/sr:.1f}s")
print(f"  chinese_lesson.wav   — {len(zh_audio)/sr:.1f}s")
print(f"  combined_output.wav  — {total_duration:.1f}s  ({file_size_kb:.1f} KB)")
