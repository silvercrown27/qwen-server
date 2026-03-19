import os
os.environ["ORT_LOGGING_LEVEL"] = "3"  # Suppress ONNX Runtime DRM/OpenCL warnings (unrelated to CUDA)

import subprocess
import time
import torch
import numpy as np
import librosa
import soundfile as sf
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel


def resolve_model(local_path: str, hf_id: str) -> str:
    """Return local model dir. Once cached, never contacts the network.
    Download order: explicit local dir → HF disk cache → network download."""
    # 1. Explicit local copy (e.g. downloaded to script dir)
    if os.path.isdir(local_path):
        return local_path
    # 2. HF disk cache — no network request
    try:
        return snapshot_download(hf_id, local_files_only=True)
    except Exception:
        pass
    # 3. Not cached at all — download once, then cache handles it forever
    print(f"  Downloading {hf_id} from HuggingFace (one-time)...")
    return snapshot_download(hf_id)

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

cv_path   = resolve_model(CV_LOCAL,   CV_HF_ID)
base_path = resolve_model(BASE_LOCAL, BASE_HF_ID)

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
    attn_impl = "flash_attention_2" if torch.cuda.is_available() else "sdpa"
except Exception:
    attn_impl = "sdpa"

# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

# Unicode ranges for scripts that require a non-English phoneme model.
# Any chunk containing at least one character from these ranges gets
# language="auto" so the model's internal code-switcher handles it,
# rather than forcing English phonetics onto foreign characters.
_FOREIGN_SCRIPT_RANGES = [
    (0x3040, 0x9FFF),   # Hiragana, Katakana, CJK Unified Ideographs (JP + ZH)
    (0xAC00, 0xD7FF),   # Hangul syllables (KO)
    (0x0400, 0x052F),   # Cyrillic (RU + others)
    (0x0600, 0x06FF),   # Arabic
]


def _has_foreign_script(text: str) -> bool:
    """True if text contains any character from a non-Latin script."""
    return any(
        lo <= ord(ch) <= hi
        for ch in text
        for lo, hi in _FOREIGN_SCRIPT_RANGES
    )


def chunk_language(text: str, primary_language: str = "english") -> str:
    """Return the language code to use for a generation call.

    - If primary_language is already a specific foreign code, use it as-is.
    - If the chunk is pure English (no foreign script), use "english".
    - If the chunk mixes English with foreign script, use "auto" so the
      model detects script boundaries internally — no audio stitching needed.
    """
    if primary_language != "english":
        return primary_language
    return "auto" if _has_foreign_script(text) else "english"


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
        # Only fade out the end of chunks that are followed by another chunk
        if i < len(normalised) - 1 and len(c) >= fade_n:
            c[-fade_n:] *= fade_out
        # Only fade in the start of chunks that follow another chunk
        if i > 0 and len(c) >= fade_n:
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
ANCHOR_TEXT = "Hello and welcome. I'm so glad you're here to learn with me today."

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
    cv_model.model = torch.compile(cv_model.model, mode="default")

    wavs, anchor_sr = cv_model.generate_custom_voice(
        text=[ANCHOR_TEXT],
        language=["english"],
        speaker=["sohee"],
        instruct=(
            "Speak at a consistent, moderate volume. Lively and animated, with a fun sense "
            "of urgency — enthusiastic delivery that makes learning feel exciting and rewarding."
        ),
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
model.model = torch.compile(model.model, mode="default")
print("Base model ready.")

print("Pre-computing voice clone prompt from anchor (x_vector_only)...")
voice_prompt = model.create_voice_clone_prompt(
    ref_audio=(anchor_wav, anchor_sr),
    x_vector_only_mode=True,
)
print("Voice prompt cached.")

# ---------------------------------------------------------------------------
# Step 3 — Lesson generation via voice cloning
# ---------------------------------------------------------------------------

BATCH_SIZE = 4


def generate_batch(batch: list[tuple[str, str]]) -> list[tuple[np.ndarray, int]]:
    """Generate multiple segments in a single GPU call.
    Safe to batch because x_vector_only_mode has no shared ICL speech tokens
    that could bleed between items (the echo artifact seen with ICL batching)."""
    texts = [t for t, _ in batch]
    langs = [l for _, l in batch]
    wavs, sr = model.generate_voice_clone(
        text=texts,
        language=langs,
        voice_clone_prompt=voice_prompt,
    )
    return [(w.astype(np.float32), sr) for w in wavs]


def ffmpeg_concat(wav_paths: list[str], output_path: str, sr: int) -> None:
    """Concatenate WAV files using FFmpeg — mirrors server's merge_audio_files_sequential()."""
    concat_list = output_path.replace(".wav", "_concat.txt")
    with open(concat_list, "w") as f:
        for p in wav_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    subprocess.run(
        ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list,
         "-c", "copy", "-y", output_path],
        check=True, capture_output=True,
    )
    os.remove(concat_list)


# ---------------------------------------------------------------------------
# LESSON 1 — Japanese
# Each chunk is a full paragraph — the model handles inline Japanese characters
# naturally. Fewer chunks = fewer generation boundaries = much less voice drift.
# ---------------------------------------------------------------------------

jp_chunks = [
    # Paragraph 1: Greetings
    ("Welcome to your Japanese lesson. Let's start with greetings. "
     "The most common way to say hello is こんにちは. "
     "In the morning you say おはようございます, meaning good morning. "
     "And in the evening, こんばんは means good evening.",
     "english"),

    # Paragraph 2: Thank you / excuse me
    ("Now let's learn some essential phrases. "
     "To say thank you, use ありがとうございます. "
     "For a casual thank you between friends, simply say ありがとう. "
     "To apologize or get someone's attention, say すみません. "
     "This works like excuse me or I'm sorry, depending on context.",
     "english"),

    # Paragraph 3: Numbers 1–5 (split into sentences — short isolated chars cause
    # phase-vocoder artefacts when bundled; one sentence per call avoids this)
    ("Now let's count from one to five.", "english"),
    ("One is いち. Two is に.", "english"),
    ("Three is さん. Four is し, or sometimes よん.", "english"),
    ("And five is ご. You might recognise those from martial arts — great work!", "english"),
]

# ---------------------------------------------------------------------------
# LESSON 2 — Mandarin Chinese
# ---------------------------------------------------------------------------

zh_chunks = [
    # Paragraph 1: Greetings
    ("Now let's explore some Mandarin Chinese. "
     "The most common greeting is 你好, which literally means you good. "
     "For a more formal hello, say 您好, where 您 is the respectful form of you. "
     "To ask how someone is doing, say 你好吗. "
     "The word 吗 turns any statement into a yes or no question.",
     "english"),

    # Paragraph 2: Food vocabulary
    ("Food is a wonderful way to connect with a language. "
     "The word for rice is 米饭. Noodles are 面条. "
     "To say I want to eat, say 我想吃. "
     "So I want to eat noodles is 我想吃面条. "
     "And if something is delicious, say 好吃, which literally means good to eat.",
     "english"),

    # Paragraph 3: The four tones (split into sentences — dense single-char tonal
    # words in one chunk cause phase-vocoder artefacts; one call per sentence avoids this)
    ("Mandarin has four tones that completely change meaning. Listen to the syllable ma in each tone.", "english"),
    ("First tone — 妈 — means mother. Second tone — 麻 — means hemp or numb.", "english"),
    ("Third tone — 马 — means horse. Fourth tone — 骂 — means to scold.", "english"),
    ("Now here is a classic tongue twister using all four: 妈妈骂马吗. Which means: does mother scold the horse? Fantastic listening!", "english"),
]

# ---------------------------------------------------------------------------
# Segments — mirrors server's intro → scene_0…N → outro structure
# Each segment is generated independently and saved as its own WAV file,
# then merged with FFmpeg — identical to AudioManager.generate_and_merge_audio_for_video()
# ---------------------------------------------------------------------------

INTRO = (
    "Welcome to PinnLab! Today we're diving into two fascinating languages — "
    "Japanese and Mandarin Chinese. Get ready to learn!"
)
OUTRO = (
    "Amazing work! You've just taken your first steps into Japanese and Mandarin. "
    "Keep practising — every word brings you closer to fluency!"
)

# Build ordered segment list: intro → JP chunks → ZH chunks → outro
# chunk_language() returns "auto" for mixed-script chunks so the model's
# internal code-switcher handles Japanese/Chinese phonetics natively.
segments: list[tuple[str, str, str]] = (
    [("intro", INTRO, "english")]
    + [(f"jp_{i}", t, chunk_language(t)) for i, (t, _) in enumerate(jp_chunks)]
    + [(f"zh_{i}", t, chunk_language(t)) for i, (t, _) in enumerate(zh_chunks)]
    + [("outro", OUTRO, "english")]
)

# ---------------------------------------------------------------------------
# Generate all segments (server-parallel: one WAV file per segment)
# ---------------------------------------------------------------------------

n_batches = (len(segments) + BATCH_SIZE - 1) // BATCH_SIZE
print(f"\n--- Generating {len(segments)} segments in {n_batches} batches (batch_size={BATCH_SIZE}) ---")
t_total = time.time()

wav_paths: list[str] = []
sr = anchor_sr

for b in range(0, len(segments), BATCH_SIZE):
    batch_meta = segments[b:b + BATCH_SIZE]
    batch_inputs = [(text, lang) for _, text, lang in batch_meta]
    labels = [label for label, _, _ in batch_meta]
    print(f"  Batch {b//BATCH_SIZE + 1}/{n_batches}: {labels}")
    t0 = time.time()
    results = generate_batch(batch_inputs)
    for (label, text, lang), (wav, sr) in zip(batch_meta, results):
        path = os.path.join(SCRIPT_DIR, f"seg_{label}.wav")
        sf.write(path, wav, sr)
        wav_paths.append(path)
        print(f"    [{label}] lang={lang!r}  {len(wav)/sr:.1f}s audio")
    print(f"    → batch done in {time.time()-t0:.1f}s")

print(f"\nAll segments generated in {time.time()-t_total:.1f}s total")

# ---------------------------------------------------------------------------
# Merge with FFmpeg — mirrors server's merge_audio_files_sequential()
# ---------------------------------------------------------------------------

output_path = os.path.join(SCRIPT_DIR, "combined_output.wav")
print(f"\nMerging {len(wav_paths)} segments with FFmpeg...")
ffmpeg_concat(wav_paths, output_path, sr)

total_duration = os.path.getsize(output_path) / (sr * 2)  # 16-bit PCM
file_size_kb = os.path.getsize(output_path) / 1024

print(f"\nSaved:")
print(f"  sohee_anchor.wav  — {len(anchor_wav)/anchor_sr:.1f}s  (voice reference)")
for p in wav_paths:
    name = os.path.basename(p)
    dur = sf.info(p).duration
    print(f"  {name:<28} — {dur:.1f}s")
print(f"  combined_output.wav — {total_duration:.1f}s  ({file_size_kb:.1f} KB)")
