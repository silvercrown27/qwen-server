import os
os.environ["ORT_LOGGING_LEVEL"] = "3"  # Suppress ONNX Runtime DRM/OpenCL warnings (unrelated to CUDA)

import time
import torch
import numpy as np
import librosa
import soundfile as sf
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CustomVoice — used only to generate the sohee reference anchor
CV_LOCAL  = os.path.join(SCRIPT_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice")
CV_HF_ID  = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Base model — supports generate_voice_clone(), used for all lesson generation
# Public model, Apache 2.0, no token required
BASE_LOCAL = os.path.join(SCRIPT_DIR, "Qwen3-TTS-12Hz-1.7B-Base")
BASE_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

cv_path   = CV_LOCAL   if os.path.isdir(CV_LOCAL)   else snapshot_download(CV_HF_ID)
base_path = BASE_LOCAL if os.path.isdir(BASE_LOCAL) else snapshot_download(BASE_HF_ID)

# ---------------------------------------------------------------------------
# CUDA diagnostics
# ---------------------------------------------------------------------------

print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = "cuda:0"
else:
    print("WARNING: CUDA not available — model will run on CPU (will be very slow)")
    device = "cpu"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    import flash_attn  # noqa: F401
    attn_impl = "flash_attention_2"
except Exception:
    attn_impl = "sdpa"

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def rms_normalize(wav: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < 1e-9:
        return wav
    return wav * (target_rms / rms)


def crossfade_join(chunks: list[np.ndarray], sr: int,
                   gap_ms: int = 70, fade_ms: int = 12) -> np.ndarray:
    """Concatenate chunks with RMS normalisation, silence gap, and linear crossfade."""
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
        if len(c) >= fade_n:
            c[-fade_n:] *= fade_out
        if len(c) >= fade_n:
            c[:fade_n] *= fade_in
        parts.append(c)
        if i < len(normalised) - 1:
            parts.append(gap)

    return np.concatenate(parts)


def pitch_normalize(wavs: list[np.ndarray], sr: int) -> list[np.ndarray]:
    """Shift each chunk's median pitch to match the first chunk's. Capped at ±3 semitones."""
    f0_medians = []
    for w in wavs:
        f0 = librosa.yin(w, fmin=60, fmax=500, sr=sr)
        voiced = f0[f0 > 0]
        f0_medians.append(float(np.median(voiced)) if len(voiced) > 0 else None)

    ref_f0 = next((f for f in f0_medians if f is not None), None)
    if ref_f0 is None:
        return wavs

    out = []
    for w, f0_mean in zip(wavs, f0_medians):
        if f0_mean is not None and f0_mean > 0 and ref_f0 > 0:
            shift = float(np.clip(12 * np.log2(ref_f0 / f0_mean), -3.0, 3.0))
            if abs(shift) > 0.1:
                w = librosa.effects.pitch_shift(w, sr=sr, n_steps=shift)
        out.append(w)
    return out

# ---------------------------------------------------------------------------
# Step 1 — Generate sohee anchor with CustomVoice model
# The anchor is cached to disk; subsequent runs skip re-generation.
# ---------------------------------------------------------------------------

ANCHOR_PATH = os.path.join(SCRIPT_DIR, "sohee_anchor.wav")
ANCHOR_TEXT = (
    "Hello and welcome. I'm so glad you're here to learn with me today. "
    "We'll explore some wonderful new words and phrases together, step by step."
)

if os.path.exists(ANCHOR_PATH):
    print(f"\nLoading cached sohee anchor: {ANCHOR_PATH}")
    anchor_wav, anchor_sr = sf.read(ANCHOR_PATH)
    anchor_wav = anchor_wav.astype(np.float32)
    print(f"  Duration: {len(anchor_wav)/anchor_sr:.1f}s")
else:
    print(f"\nGenerating sohee voice anchor with CustomVoice model...")
    print(f"  Loading: {cv_path}  (device={device}  attn={attn_impl})")
    cv_model = Qwen3TTSModel.from_pretrained(
        cv_path, device_map=device, dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    cv_model.model = torch.compile(cv_model.model, mode="reduce-overhead")

    wavs, anchor_sr = cv_model.generate_custom_voice(
        text=[ANCHOR_TEXT],
        language=["english"],
        speaker=["sohee"],
        instruct="Warm, friendly female teacher. Clear, steady, natural delivery.",
    )
    anchor_wav = wavs[0].astype(np.float32)
    sf.write(ANCHOR_PATH, anchor_wav, anchor_sr)
    print(f"  Saved: {ANCHOR_PATH}  ({len(anchor_wav)/anchor_sr:.1f}s)")

    # Free VRAM before loading the base model
    del cv_model
    torch.cuda.empty_cache()
    print("  CustomVoice model unloaded.")

# ---------------------------------------------------------------------------
# Step 2 — Load base model (supports generate_voice_clone)
# ---------------------------------------------------------------------------

print(f"\nLoading base model: {base_path}")
print(f"  device={device}  attn={attn_impl}")
model = Qwen3TTSModel.from_pretrained(
    base_path, device_map=device, dtype=torch.bfloat16,
    attn_implementation=attn_impl,
)
model.model = torch.compile(model.model, mode="reduce-overhead")
print("Base model ready.")

# ---------------------------------------------------------------------------
# Step 3 — Lesson generation via voice cloning
# ---------------------------------------------------------------------------

LESSON_INSTRUCT = "Warm, encouraging language teacher. Clear, steady pace."


def generate_lesson(chunks: list[tuple[str, str]]) -> tuple[np.ndarray, int]:
    """
    Generate all lesson chunks via generate_voice_clone() conditioned on the
    sohee anchor.  Every chunk — English and foreign words alike — inherits
    sohee's pitch, timbre, and speaking style from the reference audio.
    """
    texts = [t for t, _ in chunks]
    langs  = [l for _, l in chunks]

    print(f"  Cloning {len(texts)} chunks from sohee anchor...")
    wavs, sr = model.generate_voice_clone(
        text=texts,
        language=langs,
        ref_audio=(anchor_wav, anchor_sr),
        ref_text=ANCHOR_TEXT,
    )
    all_wavs = [w.astype(np.float32) for w in wavs]
    all_wavs = pitch_normalize(all_wavs, sr)
    return crossfade_join(all_wavs, sr), sr


# ---------------------------------------------------------------------------
# LESSON 1 — Japanese
# ---------------------------------------------------------------------------

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
# Generate lessons
# ---------------------------------------------------------------------------

print("\n--- Generating Japanese lesson ---")
t0 = time.time()
jp_audio, sr = generate_lesson(jp_chunks)
print(f"Japanese lesson: {len(jp_audio)/sr:.1f}s  ({time.time()-t0:.1f}s gen time)")
sf.write("japanese_lesson.wav", jp_audio, sr)

print("\n--- Generating Chinese lesson ---")
t0 = time.time()
zh_audio, sr = generate_lesson(zh_chunks)
print(f"Chinese lesson:  {len(zh_audio)/sr:.1f}s  ({time.time()-t0:.1f}s gen time)")
sf.write("chinese_lesson.wav", zh_audio, sr)

# ---------------------------------------------------------------------------
# Combine: Japanese → 3 s pause → Chinese
# ---------------------------------------------------------------------------

pause = np.zeros(int(sr * 3.0), dtype=np.float32)
combined = np.concatenate([jp_audio, pause, zh_audio])
sf.write("combined_output.wav", combined, sr)

total_duration = len(combined) / sr
file_size_kb = os.path.getsize("combined_output.wav") / 1024

print(f"\nSaved:")
print(f"  sohee_anchor.wav     — {len(anchor_wav)/anchor_sr:.1f}s  (voice reference)")
print(f"  japanese_lesson.wav  — {len(jp_audio)/sr:.1f}s")
print(f"  chinese_lesson.wav   — {len(zh_audio)/sr:.1f}s")
print(f"  combined_output.wav  — {total_duration:.1f}s  ({file_size_kb:.1f} KB)")
