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
ANCHOR_TEXT = "Hello and welcome. I'm so glad you're here to learn with me today!"

VOICE_INSTRUCT = (
    "Female British English presenter. Young adult to middle-aged. "
    "Bright, clear vocal texture with highly articulate and distinct pronunciation. "
    "Pitch sits in a low-to-mid female range with significant upward inflections for "
    "emphasis and excitement. Fast-paced delivery with deliberate dramatic pauses. "
    "Loud and projecting volume that increases notably during praise and announcements. "
    "Very fluent — no hesitations. Enthusiastic and excited emotion, especially when "
    "complimenting the listener. Upbeat, authoritative, and performative tone. "
    "Confident, extroverted, and engaging personality."
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

log.info("Pre-computing voice clone prompt (full ICL, x_vector_only=False)...")
t0 = time.time()
voice_prompt = model.create_voice_clone_prompt(
    ref_audio=(anchor_wav, anchor_sr),
    ref_text=ANCHOR_TEXT,
    x_vector_only_mode=False,
)
log.info(f"Voice prompt cached in {time.time()-t0:.2f}s")

# ---------------------------------------------------------------------------
# Step 3 — Segments (English only)
# ---------------------------------------------------------------------------

INTRO = (
    "Welcome to PinnLab! Today we're diving into two fascinating languages — "
    "Japanese and Mandarin Chinese. Get ready to learn!"
)
OUTRO = (
    "Amazing work! You've just taken your first steps into Japanese and Mandarin. "
    "Keep practising — every word brings you closer to fluency!"
)

segments: list[tuple[str, str]] = [
    ("intro",  INTRO),
    ("scene_0", "The law of conservation of energy states that energy cannot be created or destroyed, only transformed from one form to another."),
    ("scene_1", "Think about it this way: when you drop a ball from a height, the potential energy converts into kinetic energy as it falls."),
    ("scene_2", "When it hits the ground, that energy transforms into sound and heat. But the total amount of energy stays exactly the same!"),
    ("scene_3", "This principle applies everywhere — from the chemical energy in your breakfast to the nuclear reactions in the sun. Energy is always conserved."),
    ("outro",  OUTRO),
]

# ---------------------------------------------------------------------------
# Step 4 — Sequential generation with ICL chaining
#
# Each segment is generated one at a time. After the first, the tail of the
# previous segment's audio is fed back as the new ref_audio so the model
# inherits prosodic state (pitch, pace, energy) from the segment before it.
# This eliminates the voice-drift seam heard with independent generations.
# ---------------------------------------------------------------------------

CHAIN_TAIL_S = 2.5   # seconds of previous segment to feed as new ref_audio
SAMPLE_RATE  = anchor_sr


def tail_audio(wav: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    n = int(sr * seconds)
    return wav[-n:] if len(wav) >= n else wav


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


log.info(f"Starting sequential generation — {len(segments)} segments")
log.info(f"ICL chain tail: {CHAIN_TAIL_S}s of previous segment fed as ref for next")
t_total = time.time()

wav_paths: list[str] = []
current_prompt = voice_prompt

for i, (label, text) in enumerate(segments):
    log.info(f"[{i+1}/{len(segments)}] Segment '{label}'  ({len(text)} chars)")
    log.debug(f"  Text: {text!r}")
    t0 = time.time()

    wavs, sr = model.generate_voice_clone(
        text=text,
        language="english",
        voice_clone_prompt=current_prompt,
    )
    wav = wavs[0].astype(np.float32)
    elapsed = time.time() - t0
    duration = len(wav) / sr
    rtf = elapsed / duration if duration > 0 else 0

    path = os.path.join(SCRIPT_DIR, f"seg_{label}.wav")
    sf.write(path, wav, sr)
    wav_paths.append(path)
    log.info(f"  Done: {duration:.2f}s audio  gen={elapsed:.1f}s  RTF={rtf:.2f}x  saved={path}")

    if cuda_ok:
        mem = torch.cuda.memory_allocated() / 1024**3
        log.debug(f"  VRAM allocated: {mem:.2f} GB")

    # Chain: feed tail of this segment as ref_audio for the next
    ref_tail = tail_audio(wav, sr, CHAIN_TAIL_S)
    tail_text = text[-120:]
    log.debug(f"  Building chain prompt from {len(ref_tail)/sr:.2f}s tail  ref_text tail: {tail_text!r}")
    current_prompt = model.create_voice_clone_prompt(
        ref_audio=(ref_tail, sr),
        ref_text=tail_text,
        x_vector_only_mode=False,
    )

total_elapsed = time.time() - t_total
total_audio = sum(sf.info(p).duration for p in wav_paths)
log.info(f"All segments done — {total_audio:.1f}s audio generated in {total_elapsed:.1f}s  (RTF={total_elapsed/total_audio:.2f}x)")

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
