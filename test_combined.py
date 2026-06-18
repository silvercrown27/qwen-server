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
ANCHOR_TEXT = "Hello and welcome. I'm delighted to have you here — let's get started."

VOICE_INSTRUCT = (
    "Female British English instructor. Middle-aged, approximately 45 to 55 years old. "
    "Warm, rich, and slightly deeper female voice with a composed, measured delivery. "
    "Highly articulate and distinct pronunciation with a refined British accent. "
    "Moderately paced — deliberate and clear, never rushed, with natural pauses for emphasis. "
    "Enthusiastic and encouraging but in a calm, assured way — the warmth of a seasoned teacher "
    "who genuinely enjoys the subject, not a high-energy presenter. "
    "Steady volume with gentle lifts on key concepts and words of encouragement. "
    "Very fluent, no hesitations. Authoritative yet approachable tone. "
    "Mature, confident, and intellectually engaging personality."
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

# x_vector_only=True: encodes only the speaker embedding — no ICL speech tokens.
# This prompt is built once and reused for every segment, avoiding the per-segment
# ref-audio re-encoding cost of full ICL mode.
log.info("Pre-computing voice clone prompt (x_vector_only=True)...")
t0 = time.time()
voice_prompt = model.create_voice_clone_prompt(
    ref_audio=(anchor_wav, anchor_sr),
    x_vector_only_mode=True,
)
log.info(f"Voice prompt cached in {time.time()-t0:.2f}s")

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
# Step 4 — Sequential generation
#
# x_vector prompt is reused for every segment (no re-encoding).
# max_new_tokens is capped per segment based on text length so short segments
# don't burn the full 2048-token budget waiting for an EOS that never comes.
# Approx 12 codec tokens/s × estimated speech rate of ~14 chars/s → ~0.86 tok/char.
# We add a 50% headroom buffer and clamp to [256, 2048].
# ---------------------------------------------------------------------------

TOKENS_PER_CHAR = 0.86
TOKEN_HEADROOM   = 1.5
TOKEN_MIN        = 256
TOKEN_MAX        = 2048


def tokens_for_text(text: str) -> int:
    estimate = int(len(text) * TOKENS_PER_CHAR * TOKEN_HEADROOM)
    return max(TOKEN_MIN, min(TOKEN_MAX, estimate))


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


log.info(f"Starting generation — {len(segments)} segments  (x_vector prompt, scaled max_new_tokens)")
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
        voice_clone_prompt=voice_prompt,
        max_new_tokens=max_tok,
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
