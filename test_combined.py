import os
os.environ["ORT_LOGGING_LEVEL"] = "3"  # Suppress ONNX Runtime DRM/OpenCL warnings

import subprocess
import time
import logging
import torch
import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tts")


def resolve_model(local_path: str, hf_id: str) -> str:
    if os.path.isdir(local_path):
        log.debug(f"Model found locally: {local_path}")
        return local_path
    try:
        path = snapshot_download(hf_id, local_files_only=True)
        log.debug(f"Model found in HF cache: {path}")
        return path
    except Exception:
        pass
    log.info(f"Downloading {hf_id} from HuggingFace (one-time)...")
    return snapshot_download(hf_id)


# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VD_LOCAL  = os.path.join(SCRIPT_DIR, "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
VD_HF_ID  = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

BASE_LOCAL = os.path.join(SCRIPT_DIR, "Qwen3-TTS-12Hz-1.7B-Base")
BASE_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

log.info("Resolving model paths...")
vd_path   = resolve_model(VD_LOCAL,   VD_HF_ID)
base_path = resolve_model(BASE_LOCAL, BASE_HF_ID)
log.info(f"VoiceDesign path : {vd_path}")
log.info(f"Base model path  : {base_path}")

# ---------------------------------------------------------------------------
# CUDA diagnostics
# ---------------------------------------------------------------------------

cuda_ok = torch.cuda.is_available()
log.info(f"CUDA available : {cuda_ok}")
if cuda_ok:
    log.info(f"GPU            : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log.info(f"VRAM           : {vram:.1f} GB")
    device = "cuda:0"
else:
    log.warning("CUDA not available — running on CPU (will be slow)")
    device = "cpu"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    import flash_attn  # noqa: F401
    attn_impl = "flash_attention_2" if cuda_ok else "sdpa"
    log.info("flash-attn found — using flash_attention_2")
except Exception:
    attn_impl = "sdpa"
    log.info("flash-attn not found — using sdpa")

# ---------------------------------------------------------------------------
# Voice design config
# ---------------------------------------------------------------------------

ANCHOR_PATH = os.path.join(SCRIPT_DIR, "host_anchor.wav")
ANCHOR_TEXT = (
    "Good morning, everyone. Before we begin, I want you to know that the concepts "
    "we cover today will fundamentally change how you see the world around you. "
    "I have been teaching this for over twenty years, and it still fascinates me."
)

VOICE_INSTRUCT = (
    "A 52-year-old British female university professor with over two decades of teaching experience. "
    "Her voice is deep for a woman — a low contralto, with strong chest resonance and no breathiness. "
    "Fundamental frequency around 165 to 180 Hz, noticeably lower than a young woman's voice. "
    "Her pitch range is moderate — she uses expressive upward inflections on key words and moments of "
    "excitement, but always returns to a low, grounded baseline. "
    "Pace is slow and deliberate — around 110 words per minute. She never rushes. "
    "She takes natural breath pauses between clauses and allows each idea to land fully before moving on. "
    "Longer pauses after key concepts give the listener time to absorb what was just said. "
    "Articulation is precise and crisp — received pronunciation British English. "
    "Her enthusiasm is vivid and infectious — she genuinely loves the subject and lets it show, "
    "leaning into exciting ideas with increased energy and a brighter tone, like a professor who "
    "still gets a spark of joy every time she explains a concept she finds beautiful. "
    "Statements end with falling intonation — no upspeak, no vocal fry. "
    "Her personality is warm, intellectually passionate, authoritative, and deeply engaging."
)

# ---------------------------------------------------------------------------
# Step 1 — Generate anchor with VoiceDesign model (cached after first run)
# ---------------------------------------------------------------------------

if os.path.exists(ANCHOR_PATH):
    log.info(f"Loading cached host anchor: {ANCHOR_PATH}")
    t0 = time.time()
    anchor_wav, anchor_sr = sf.read(ANCHOR_PATH)
    anchor_wav = anchor_wav.astype(np.float32)
    log.info(f"  Loaded {len(anchor_wav)/anchor_sr:.2f}s audio  @ {anchor_sr} Hz  ({time.time()-t0:.2f}s)")
else:
    log.info("No cached anchor found — generating with VoiceDesign model")
    log.info(f"  Model : {vd_path}")
    log.info(f"  Device: {device}  attn: {attn_impl}")
    log.info(f"  Anchor text ({len(ANCHOR_TEXT)} chars): {ANCHOR_TEXT!r}")
    log.info(f"  Voice instruct ({len(VOICE_INSTRUCT)} chars): {VOICE_INSTRUCT!r}")
    t_load = time.time()
    vd_model = Qwen3TTSModel.from_pretrained(
        vd_path, device_map=device, dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    log.info(f"  VoiceDesign model loaded in {time.time()-t_load:.1f}s")
    if cuda_ok:
        mem = torch.cuda.memory_allocated() / 1024**3
        log.debug(f"  VRAM allocated after load: {mem:.2f} GB")

    log.info("  Calling generate_voice_design()...")
    t_gen = time.time()
    wavs, anchor_sr = vd_model.generate_voice_design(
        text=ANCHOR_TEXT,
        language="english",
        instruct=VOICE_INSTRUCT,
    )
    anchor_wav = wavs[0].astype(np.float32)
    log.info(f"  Generated {len(anchor_wav)/anchor_sr:.2f}s audio in {time.time()-t_gen:.1f}s  @ {anchor_sr} Hz")
    sf.write(ANCHOR_PATH, anchor_wav, anchor_sr)
    log.info(f"  Saved anchor: {ANCHOR_PATH}")
    del vd_model
    torch.cuda.empty_cache()
    if cuda_ok:
        mem = torch.cuda.memory_allocated() / 1024**3
        log.debug(f"  VRAM after unload: {mem:.2f} GB")
    log.info("  VoiceDesign model unloaded.")

# ---------------------------------------------------------------------------
# Step 2 — Load base model for voice cloning
# ---------------------------------------------------------------------------

log.info(f"Loading base model: {base_path}")
log.info(f"  device={device}  attn={attn_impl}")
t_load = time.time()
model = Qwen3TTSModel.from_pretrained(
    base_path, device_map=device, dtype=torch.bfloat16,
    attn_implementation=attn_impl,
)
log.info(f"Base model loaded in {time.time()-t_load:.1f}s")
if cuda_ok:
    mem = torch.cuda.memory_allocated() / 1024**3
    log.debug(f"  VRAM allocated: {mem:.2f} GB")

log.info("Base model ready.")

# ---------------------------------------------------------------------------
# Step 3 — Segments (English only)
# ---------------------------------------------------------------------------

INTRO = (
    "Welcome to PinnLab. Today we're exploring one of the most elegant principles in all of physics — "
    "the law of conservation of energy. I think you'll find it quite beautiful."
)
OUTRO = (
    "Well done. You've covered the core idea and seen it applied across several real-world contexts. "
    "Keep reflecting on these examples — they'll deepen your understanding every time. See you in the next lesson."
)

segments: list[tuple[str, str]] = [
    ("intro",   INTRO),
    ("scene_0", "The law of conservation of energy states that energy cannot be created or destroyed — only transformed from one form to another."),
    ("scene_1", "Consider a ball dropped from a height. At the top, it holds potential energy. As it falls, that potential energy converts into kinetic energy — the energy of motion."),
    ("scene_2", "When the ball strikes the ground, that kinetic energy doesn't vanish. It transforms into sound and heat. The total, at every point, remains exactly the same."),
    ("scene_3", "This principle operates everywhere — from the chemical energy in your breakfast becoming the mechanical energy that moves your muscles, to the nuclear reactions in our sun radiating light and warmth across the solar system."),
    ("outro",   OUTRO),
]

# ---------------------------------------------------------------------------
# Step 4 — Single-pass generation using voice_anchor.wav
#
# voice_anchor.wav is the saved outro from a previous run — the voice at its
# most settled, natural state. Every segment is cloned from it using full ICL
# mode (x_vector_only=False) so the model has real speech tokens to match
# pitch, pace, and timbre against. The same prompt is reused for all segments
# — no chaining, no drift.
#
# If voice_anchor.wav does not exist (first run), we fall back to
# host_anchor.wav (VoiceDesign output) and generate normally. The outro from
# that run is saved as voice_anchor.wav so all future runs use it.
# ---------------------------------------------------------------------------

TOKENS_PER_CHAR    = 0.86
TOKEN_HEADROOM     = 1.5
TOKEN_MIN          = 256
TOKEN_MAX          = 2048

VOICE_ANCHOR_PATH  = os.path.join(SCRIPT_DIR, "voice_anchor.wav")
OUTRO_TEXT         = segments[-1][1]


def tokens_for_text(text: str) -> int:
    estimate = int(len(text) * TOKENS_PER_CHAR * TOKEN_HEADROOM)
    return max(TOKEN_MIN, min(TOKEN_MAX, estimate))


CODEC_HZ   = 12          # speech tokenizer frame rate
FADE_MS    = 30          # fade-in/out applied to ref audio edges


def prepare_ref_audio(wav: np.ndarray, sr: int) -> np.ndarray:
    """Trim to codec frame boundary and apply edge fades to eliminate static.

    The codec tokenizer runs at CODEC_HZ. Any samples beyond the last complete
    frame produce a malformed partial token which decodes as static noise.
    Trimming to the nearest frame boundary and fading the edges prevents this.
    """
    # Trim to nearest complete codec frame
    frame_samples = sr // CODEC_HZ
    n_frames = len(wav) // frame_samples
    trimmed = wav[:n_frames * frame_samples]

    # Short linear fade-in and fade-out to avoid onset/offset clicks
    fade_n = int(sr * FADE_MS / 1000)
    fade_n = min(fade_n, len(trimmed) // 4)
    if fade_n > 0:
        trimmed = trimmed.copy()
        trimmed[:fade_n]  *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        trimmed[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)

    removed = len(wav) - len(trimmed)
    log.debug(f"  prepare_ref_audio: {len(wav)} → {len(trimmed)} samples  "
              f"(trimmed {removed}, fade {fade_n} samples each end)")
    return trimmed


def ffmpeg_concat(wav_paths: list[str], output_path: str) -> None:
    concat_list = output_path.replace(".wav", "_concat.txt")
    with open(concat_list, "w") as f:
        for p in wav_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    log.debug(f"  FFmpeg concat list: {concat_list}  ({len(wav_paths)} files)")
    result = subprocess.run(
        ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list,
         "-c", "copy", "-y", output_path],
        capture_output=True,
    )
    os.remove(concat_list)
    if result.returncode != 0:
        log.error(f"FFmpeg failed:\n{result.stderr.decode()}")
        result.check_returncode()
    log.debug(f"  FFmpeg exit code: {result.returncode}")


if os.path.exists(VOICE_ANCHOR_PATH):
    log.info(f"Loading voice anchor: {VOICE_ANCHOR_PATH}")
    ref_wav, ref_sr = sf.read(VOICE_ANCHOR_PATH)
    ref_wav = ref_wav.astype(np.float32)
    ref_text = OUTRO_TEXT
    log.info(f"  {len(ref_wav)/ref_sr:.2f}s @ {ref_sr} Hz")
else:
    log.info("voice_anchor.wav not found — first run, using host_anchor.wav")
    log.info("voice_anchor.wav will be created from the outro of this run.")
    ref_wav, ref_sr = anchor_wav, anchor_sr
    ref_text = ANCHOR_TEXT

ref_wav = prepare_ref_audio(ref_wav, ref_sr)
log.info(f"  Ref audio prepared: {len(ref_wav)/ref_sr:.2f}s (frame-aligned, faded)")

log.info("Building ICL voice clone prompt (x_vector_only=False)...")
t0 = time.time()
clone_prompt = model.create_voice_clone_prompt(
    ref_audio=(ref_wav, ref_sr),
    ref_text=ref_text,
    x_vector_only_mode=False,
)
log.info(f"Clone prompt built in {time.time()-t0:.2f}s")

log.info(f"Starting generation — {len(segments)} segments")
t_total = time.time()

wav_paths: list[str] = []

for i, (label, text) in enumerate(segments):
    max_tok = tokens_for_text(text)
    log.info(f"[{i+1}/{len(segments)}] Segment '{label}'  ({len(text)} chars  max_new_tokens={max_tok})")
    log.debug(f"  Text: {text!r}")
    t0 = time.time()

    wavs, sr = model.generate_voice_clone(
        text=text,
        language="english",
        voice_clone_prompt=clone_prompt,
        max_new_tokens=max_tok,
        temperature=0.7,
        repetition_penalty=1.02,
    )
    wav = wavs[0].astype(np.float32)
    elapsed = time.time() - t0
    duration = len(wav) / sr
    log.info(f"  Done: {duration:.2f}s audio  gen={elapsed:.1f}s  RTF={elapsed/duration:.2f}x")

    path = os.path.join(SCRIPT_DIR, f"seg_{label}.wav")
    sf.write(path, wav, sr)
    wav_paths.append(path)

    if cuda_ok:
        mem = torch.cuda.memory_allocated() / 1024**3
        log.debug(f"  VRAM allocated: {mem:.2f} GB")

# Save the outro as voice_anchor.wav for all future runs
outro_path = os.path.join(SCRIPT_DIR, "seg_outro.wav")
outro_wav, outro_sr = sf.read(outro_path)
sf.write(VOICE_ANCHOR_PATH, outro_wav, outro_sr)
log.info(f"voice_anchor.wav updated from seg_outro.wav  ({len(outro_wav)/outro_sr:.2f}s)")

total_elapsed = time.time() - t_total
total_audio = sum(sf.info(p).duration for p in wav_paths)
log.info(f"All segments done — {total_audio:.1f}s audio in {total_elapsed:.1f}s  (RTF={total_elapsed/total_audio:.2f}x)")

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

output_path = os.path.join(SCRIPT_DIR, "combined_output.wav")
log.info(f"Merging {len(wav_paths)} segments → {output_path}")
t0 = time.time()
ffmpeg_concat(wav_paths, output_path)
log.info(f"Merge done in {time.time()-t0:.2f}s")

log.info("=== Output summary ===")
log.info(f"  host_anchor.wav   {len(anchor_wav)/anchor_sr:.2f}s  (voice reference)")
for p in wav_paths:
    log.info(f"  {os.path.basename(p):<32} {sf.info(p).duration:.2f}s")
out_size_kb = os.path.getsize(output_path) / 1024
log.info(f"  combined_output.wav  {out_size_kb:.1f} KB")
