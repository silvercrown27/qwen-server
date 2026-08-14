"""
Decisive A/B: x_vector_only_mode=True vs. False, BOTH using a
LANGUAGE-MATCHED reference anchor (not the English-only one).

WHY THIS TEST EXISTS: test_dual_reference_jp.py's dual-reference approach
used FULL ICL (x_vector_only_mode=False) with a VoiceDesign-generated
Japanese anchor for Celeste. That fixed the "unbearable" Japanese from
test_hotel_scenario_final.py's Run C — but Run C's xvec test used Celeste's
ENGLISH reference (celeste.wav) to extract the speaker embedding, then
asked it to speak Japanese. That's a confounded comparison: xvec mode
wasn't necessarily bad at Japanese — it was bad at Japanese GIVEN an
English-only reference. This script re-tests xvec FAIRLY, using the SAME
Japanese-native anchor the ICL run already validated, to see whether xvec
+ matched-anchor closes the gap.

WHY THIS MATTERS FOR THE PRODUCTION DESIGN: if xvec+matched-anchor sounds
comparable to ICL+matched-anchor, xvec is strictly preferable for any
content that needs more than one language per voice (which, per the wider
multilingual rollout, is now every voice) — no accent-bleed risk (per
QwenLM/Qwen3-TTS discussion #230, xvec is explicitly cited as immune to
this since it carries no acoustic/phonetic context, only a speaker
embedding), simpler to reason about (a fixed ~10-token prefill regardless
of reference length, vs. ICL's variable ref_code context), and per
community-measured numbers, meaningfully faster (x-vector-only requires
only ~10 tokens of prefill vs. ICL's 100+).

WHAT THIS SCRIPT DOES:
  1. Generates (or reuses cached) Celeste JP anchor via VoiceDesign — same
     as test_dual_reference_jp.py, using scene/voice_language_profiles.py's
     instruct/anchor_text for (celeste, japanese) as the single source of
     truth for this content, rather than duplicating it inline.
  2. Builds TWO clone prompts from that SAME anchor: one ICL
     (x_vector_only_mode=False), one xvec (x_vector_only_mode=True).
  3. Runs the same mixed EN/JP scenario against both.
  4. Also runs both modes against ENGLISH-only content (celeste.wav) as a
     sanity check — xvec's quality tradeoff on English-only text (no
     language-switching involved) should be small; if it's large even
     there, that changes the recommendation for ALL content, not just
     multilingual chunks.

Run (GPU instance, same venv as qwen-server):
    cd qwen-server && source venv/bin/activate
    python test_matched_anchor_mode_ab.py

Output: matched_anchor_ab_out/{icl_jp_anchor,xvec_jp_anchor,
icl_en_anchor,xvec_en_anchor}/*.wav — listen to the _jp_anchor pair first
(the actual open question), then the _en_anchor pair as a sanity check.
"""
import os
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

import sys
import time
import logging
import subprocess
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("matched_anchor_ab")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "matched_anchor_ab_out")
os.makedirs(OUT_DIR, exist_ok=True)

BASE_LOCAL = os.path.join(SCRIPT_DIR, "Qwen3-TTS-12Hz-1.7B-Base")
BASE_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VD_LOCAL = os.path.join(SCRIPT_DIR, "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
VD_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

_MAIN_APP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "pinn-research-model"))
if _MAIN_APP_DIR not in sys.path:
    sys.path.insert(0, _MAIN_APP_DIR)
from scene.voice_language_profiles import VOICE_LANGUAGE_PROFILES  # noqa: E402

CELESTE_EN_REF_WAV = os.path.join(_MAIN_APP_DIR, "scene", "voices", "celeste.wav")
CELESTE_EN_REF_TEXT = (
    "Good evening. I have spent many years exploring this subject, and what "
    "continues to move me is how much of it still surprises me. I hope that "
    "by the end of today, it surprises you too."
)
CELESTE_JP_PROFILE = VOICE_LANGUAGE_PROFILES["celeste"]["japanese"]

AUDIO_PAD_SECONDS = 0.1
REF_TRAILING_SILENCE_S = 0.5
GENERATION_SEED = 42
CODEC_HZ = 12
FADE_MS = 30


def resolve_model(local_path: str, hf_id: str) -> str:
    if os.path.isdir(local_path):
        return local_path
    try:
        return snapshot_download(hf_id, local_files_only=True)
    except Exception:
        pass
    log.info(f"Downloading {hf_id} from HuggingFace (one-time)...")
    return snapshot_download(hf_id)


def prepare_ref_audio(wav: np.ndarray, sr: int, trailing_silence_s: float = REF_TRAILING_SILENCE_S) -> np.ndarray:
    frame_samples = sr // CODEC_HZ
    n_frames = len(wav) // frame_samples
    trimmed = wav[: n_frames * frame_samples].copy()
    fade_n = min(int(sr * FADE_MS / 1000), len(trimmed) // 4)
    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    trimmed[:fade_n] *= ramp
    trimmed[-fade_n:] *= ramp[::-1]
    trimmed = trimmed.astype(np.float32)
    if trailing_silence_s > 0:
        silence = np.zeros(int(sr * trailing_silence_s), dtype=np.float32)
        trimmed = np.concatenate([trimmed, silence])
    return trimmed


def pad_audio(wav: np.ndarray, sr: int, pad_seconds: float = AUDIO_PAD_SECONDS) -> np.ndarray:
    silence = np.zeros(int(sr * pad_seconds), dtype=wav.dtype)
    return np.concatenate([silence, wav, silence])


def tokens_for_text(text: str) -> int:
    return max(192, min(2048, int(len(text) * 0.86 * 1.5)))


# Same mixed EN/JP scenario as prior scripts, kept identical for direct
# comparability across this whole investigation.
JP_SCENARIO: list[tuple[str, str]] = [
    ("b00",
     "Okay, we've just landed in Tokyo — and the very first thing you'll hear at the front "
     "desk is こんにちは、予約をしております, which simply means 'hello, I have a reservation.'"),
    ("b01",
     "If they ask to see your passport, you'll hear パスポートを拝見してもよろしいでしょうか — "
     "and don't worry, they're just politely asking to take a look at it."),
    ("b02",
     "They might also mention 朝食は一階のレストランで、七時から十時までとなっております, which "
     "tells you breakfast is served downstairs from seven until ten every morning."),
    ("b03",
     "Later, if you need directions, a simple wave and エレベーターはあちらにございます means "
     "'the elevator is right over there' — short, polite, and easy to catch once you know it."),
]

# English-only sanity-check content — no language switching, just tests
# whether xvec's known fidelity/emotion cost is noticeable even without
# any cross-lingual pronunciation question involved.
EN_SCENARIO: list[tuple[str, str]] = [
    ("e00", "Okay, we've just landed in Tokyo, and our first stop is the hotel front desk."),
    ("e01", "Let's get you ready with a few essential phrases before we walk in — don't worry, "
            "this is going to be easier than you think!"),
    ("e02", "Once you're checked in, you'll want to find the elevator and maybe a nearby "
            "convenience store — trust me, you'll need both within the hour."),
]


def load_base_model():
    cuda_ok = torch.cuda.is_available()
    device = "cuda:0" if cuda_ok else "cpu"
    dtype = torch.bfloat16 if cuda_ok else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2" if cuda_ok else "sdpa"
    except Exception:
        attn_impl = "sdpa"
    if not cuda_ok:
        log.warning("No CUDA device detected — this script is designed for a GPU box.")
    log.info(f"Device={device}  dtype={dtype}  attn={attn_impl}")

    base_path = resolve_model(BASE_LOCAL, BASE_HF_ID)
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        base_path, device_map=device, dtype=dtype, attn_implementation=attn_impl,
    )
    log.info(f"Base model loaded in {time.time()-t0:.1f}s")
    return model, device, dtype, attn_impl


def get_or_generate_jp_anchor(device, dtype, attn_impl) -> str:
    anchor_path = os.path.join(OUT_DIR, "celeste_jp_anchor.wav")
    if os.path.exists(anchor_path):
        log.info(f"Using cached JP anchor: {anchor_path}")
        return anchor_path

    vd_path = resolve_model(VD_LOCAL, VD_HF_ID)
    log.info(f"Loading VoiceDesign model: {vd_path}...")
    vd_model = Qwen3TTSModel.from_pretrained(
        vd_path, device_map=device, dtype=dtype, attn_implementation=attn_impl,
    )
    log.info("Generating Japanese-native Celeste-character anchor...")
    wavs, sr = vd_model.generate_voice_design(
        text=CELESTE_JP_PROFILE["anchor_text"],
        language="Japanese",
        instruct=CELESTE_JP_PROFILE["instruct"],
    )
    anchor_wav = wavs[0].astype(np.float32)
    sf.write(anchor_path, anchor_wav, sr)
    log.info(f"  Saved: {anchor_path} ({len(anchor_wav)/sr:.2f}s)")

    del vd_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return anchor_path


def build_prompt(model, ref_wav_path: str, ref_text: str, x_vector_only_mode: bool):
    ref_wav, ref_sr = sf.read(ref_wav_path)
    ref_wav = prepare_ref_audio(ref_wav.astype(np.float32), ref_sr)
    if x_vector_only_mode:
        return model.create_voice_clone_prompt(ref_audio=(ref_wav, ref_sr), x_vector_only_mode=True)
    return model.create_voice_clone_prompt(ref_audio=(ref_wav, ref_sr), ref_text=ref_text, x_vector_only_mode=False)


def run_scenario(model, prompt, run_name: str, scenario: list[tuple[str, str]]) -> str:
    run_dir = os.path.join(OUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"=== RUN '{run_name}' ===")
    paths = []
    for chunk_id, text in scenario:
        torch.manual_seed(GENERATION_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(GENERATION_SEED)
        t0 = time.time()
        wavs, sr = model.generate_voice_clone(
            text=text, language="auto", voice_clone_prompt=prompt,
            max_new_tokens=tokens_for_text(text), temperature=0.7, repetition_penalty=1.05,
        )
        elapsed = time.time() - t0
        padded = pad_audio(wavs[0].astype(np.float32), sr)
        path = os.path.join(run_dir, f"{chunk_id}.wav")
        sf.write(path, padded, sr)
        paths.append(path)
        log.info(f"  {chunk_id}: {len(padded)/sr:.2f}s audio in {elapsed:.1f}s gen time")

    merged_path = os.path.join(run_dir, f"_merged_{run_name}.wav")
    concat_list = merged_path.replace(".wav", "_concat.txt")
    with open(concat_list, "w") as f:
        for p in paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    subprocess.run(
        ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", "-y", merged_path],
        capture_output=True, check=True,
    )
    os.remove(concat_list)
    log.info(f"  Merged: {merged_path}")
    return merged_path


def main():
    model, device, dtype, attn_impl = load_base_model()

    if not os.path.exists(CELESTE_EN_REF_WAV):
        raise FileNotFoundError(f"Celeste English reference not found: {CELESTE_EN_REF_WAV}")

    jp_anchor_path = get_or_generate_jp_anchor(device, dtype, attn_impl)

    log.info("Building 4 clone prompts: {ICL,xvec} x {JP-anchor,EN-anchor}...")
    icl_jp_prompt = build_prompt(model, jp_anchor_path, CELESTE_JP_PROFILE["anchor_text"], x_vector_only_mode=False)
    xvec_jp_prompt = build_prompt(model, jp_anchor_path, "", x_vector_only_mode=True)
    icl_en_prompt = build_prompt(model, CELESTE_EN_REF_WAV, CELESTE_EN_REF_TEXT, x_vector_only_mode=False)
    xvec_en_prompt = build_prompt(model, CELESTE_EN_REF_WAV, "", x_vector_only_mode=True)

    icl_jp_merged = run_scenario(model, icl_jp_prompt, "icl_jp_anchor", JP_SCENARIO)
    xvec_jp_merged = run_scenario(model, xvec_jp_prompt, "xvec_jp_anchor", JP_SCENARIO)
    icl_en_merged = run_scenario(model, icl_en_prompt, "icl_en_anchor", EN_SCENARIO)
    xvec_en_merged = run_scenario(model, xvec_en_prompt, "xvec_en_anchor", EN_SCENARIO)

    log.info("=" * 90)
    log.info("RESULTS — listen in this order:")
    log.info(f"  [1] icl_jp_anchor   (the current best-known-good):  {icl_jp_merged}")
    log.info(f"  [2] xvec_jp_anchor  (THE question this test answers): {xvec_jp_merged}")
    log.info(f"  [3] icl_en_anchor   (English sanity check, ICL):    {icl_en_merged}")
    log.info(f"  [4] xvec_en_anchor  (English sanity check, xvec):   {xvec_en_merged}")
    log.info(
        "\nDECISION RULE:\n"
        "  - If [2] sounds as clean/natural in Japanese as [1] (or close enough that the\n"
        "    difference doesn't bother a fluent Japanese listener) -> USE XVEC for all\n"
        "    multilingual content going forward: no accent-bleed risk, faster, simpler.\n"
        "  - If [2] is still noticeably worse than [1] in Japanese -> the earlier 'xvec is\n"
        "    unbearable' result wasn't just about the English-only reference — ICL is\n"
        "    genuinely necessary for cross-lingual cloning quality, accept the accent-bleed\n"
        "    risk as the lesser problem.\n"
        "  - Compare [3] vs [4] separately: if xvec sounds noticeably flatter/less\n"
        "    expressive even on plain English (no language-switching involved), that's\n"
        "    the ALREADY-KNOWN fidelity cost of x_vector_only_mode showing up regardless\n"
        "    of the multilingual question — factor that into whether xvec-everywhere is\n"
        "    worth it even if [2] sounds fine."
    )


if __name__ == "__main__":
    main()
