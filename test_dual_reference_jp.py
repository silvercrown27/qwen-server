"""
Dual-reference test — Celeste (English) + a VoiceDesign-generated
Japanese-native counterpart, switched per-chunk by dominant language.

WHY THIS EXISTS: test_hotel_scenario_final.py's Run B (full ICL) and Run C
(x-vector-only) both used Celeste's ENGLISH-ONLY reference clip for
Japanese-containing chunks. Real GPU listening results: x-vector-only's
Japanese was "unbearable," and full ICL's Japanese was better but still
"English sounding" — bad enough to bother a fluent Japanese listener, in
BOTH modes. External research explains why neither mode-level parameter
fixes this: "Qwen3-TTS tends to fall back to its dominant English
distribution unless conditioned on a real example of the target language"
— the model has never heard THIS specific voice produce a Japanese
phoneme, so no amount of language="auto"/"japanese" forcing changes what
acoustic reference it's extrapolating from. The community's own stated fix
is exactly this: "voice cloning with a reference clip recorded in the
target language."

APPROACH: rather than picking an unrelated off-the-shelf Japanese speaker
(Qwen3-TTS-CustomVoice's built-in "Ono_Anna" would work but sounds nothing
like Celeste — a language switch would sound like a different person, not
just a different accent), this uses the VoiceDesign model — already a
first-class dependency in this codebase (see tts_service/server.py's
_worker_init, which downloads it on-demand for exactly this purpose) — to
GENERATE a new Japanese-native anchor clip described with the SAME
character traits as Celeste's existing English instruct string (gentle,
soothing, calm, patient-teacher pace). This keeps voice CHARACTER
consistent across the language switch even though the underlying acoustic
reference is a different recording. A community writeup independently
validated this exact workflow ("creates a reference voice with VoiceDesign
and then clones it with the Base model for reuse") for Japanese quality
specifically.

RESULT: two clone prompts for Celeste —
  celeste_en_prompt  <- celeste.wav (existing, English)
  celeste_jp_prompt  <- newly VoiceDesign-generated Japanese anchor

A chunk picks whichever prompt matches its DOMINANT language rather than
using one prompt for everything. This is the same "per-chunk language
resolution" pattern audio_manager._resolve_tts_language already does for
the `language=` parameter — extended here to also resolve which REFERENCE
PROMPT to use, not just which language string to pass.

Run (GPU instance, same venv as qwen-server):
    cd qwen-server && source venv/bin/activate
    python test_dual_reference_jp.py

Output: dual_ref_test_out/{celeste_jp_anchor.wav, run_single_ref_en/*.wav,
run_dual_ref/*.wav}, plus a diagnostic report comparing single-reference
(today's approach) against dual-reference (this test) on the SAME Japanese-
heavy content.
"""
import os
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

import time
import logging
import subprocess
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("dual_ref_test")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "dual_ref_test_out")
os.makedirs(OUT_DIR, exist_ok=True)

BASE_LOCAL = os.path.join(SCRIPT_DIR, "Qwen3-TTS-12Hz-1.7B-Base")
BASE_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VD_LOCAL = os.path.join(SCRIPT_DIR, "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
VD_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

CELESTE_REF_WAV = os.path.join(
    SCRIPT_DIR, "..", "pinn-research-model", "scene", "voices", "celeste.wav"
)
CELESTE_REF_TEXT = (
    "Good evening. I have spent many years exploring this subject, and what "
    "continues to move me is how much of it still surprises me. I hope that "
    "by the end of today, it surprises you too."
)

# Mirrors Celeste's existing English instruct string (scene/qwen_voice.py's
# VOICE_CONFIGS["celeste"]["instruct"]) as closely as natural Japanese
# phrasing allows, so VoiceDesign targets the SAME character — gentle,
# soothing, calm/patient pace — just realized as a native Japanese speaker
# rather than translating/forcing the English-trained voice into Japanese.
CELESTE_JP_INSTRUCT = (
    "穏やかで優しい日本語の女性の声。落ち着いた、安心感のある話し方 — "
    "決して急がない、忍耐強い先生のような口調。聞き手がじっくり理解できるよう、"
    "複雑な話題にも適した、温かみのある声。"
)
# Natural, moderate-length Japanese anchor sentence for VoiceDesign to speak
# — analogous in length/register to Celeste's English anchor_text
# ("Hello there. Let's take a moment to really understand this together.")
CELESTE_JP_ANCHOR_TEXT = "こんにちは。今日は一緒に、じっくりと理解を深めていきましょう。"

AUDIO_PAD_SECONDS = 0.1
REF_TRAILING_SILENCE_S = 0.5
GENERATION_SEED = 42


def resolve_model(local_path: str, hf_id: str) -> str:
    if os.path.isdir(local_path):
        return local_path
    try:
        return snapshot_download(hf_id, local_files_only=True)
    except Exception:
        pass
    log.info(f"Downloading {hf_id} from HuggingFace (one-time)...")
    return snapshot_download(hf_id)


CODEC_HZ = 12
FADE_MS = 30


def prepare_ref_audio(wav: np.ndarray, sr: int, trailing_silence_s: float = REF_TRAILING_SILENCE_S) -> np.ndarray:
    """Codec-frame-align + edge-fade + append trailing silence (fix for the
    ICL leaked-fragment artifact — see test_hotel_scenario_final.py's
    prepare_ref_audio docstring for the full external-research citation)."""
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


def strip_leading_reference_leak(wav: np.ndarray, sr: int) -> tuple:
    """Relative dip-then-louder-recovery detector — see
    test_hotel_scenario_final.py for the full derivation/validation of this
    heuristic against real generated audio. Kept as a secondary safety net
    on top of the silence-append fix in prepare_ref_audio."""
    LEAK_SCAN_WINDOW_S, LEAK_FRAME_S = 0.30, 0.01
    LEAK_DIP_RATIO, LEAK_MIN_GAP_S, LEAK_RECOVERY_RATIO = 0.5, 0.02, 1.15
    RECOVERY_LOOKAHEAD_FRAMES = 5

    frame_n = int(sr * LEAK_FRAME_S)
    scan_frames = int(LEAK_SCAN_WINDOW_S / LEAK_FRAME_S)
    min_gap_frames = max(1, int(LEAK_MIN_GAP_S / LEAK_FRAME_S))
    frame_rms = []
    for i in range(scan_frames):
        seg = wav[i * frame_n:(i + 1) * frame_n]
        if len(seg) == 0:
            break
        frame_rms.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))))
    if not frame_rms:
        return wav, 0.0

    burst_peak = frame_rms[0]
    i = 1
    while i < len(frame_rms):
        burst_peak = max(burst_peak, frame_rms[i])
        if frame_rms[i] <= burst_peak * LEAK_DIP_RATIO:
            gap_start = i
            j = i
            while j < len(frame_rms) and frame_rms[j] <= burst_peak * LEAK_DIP_RATIO:
                j += 1
            gap_len = j - gap_start
            recovery_window = frame_rms[j:j + RECOVERY_LOOKAHEAD_FRAMES]
            recovery_peak = max(recovery_window) if recovery_window else 0.0
            if gap_len >= min_gap_frames and j < len(frame_rms) and recovery_peak >= burst_peak * LEAK_RECOVERY_RATIO:
                cut_sample = j * frame_n
                return wav[cut_sample:], cut_sample / sr
            i = j if j > i else i + 1
            continue
        i += 1
    return wav, 0.0


def tokens_for_text(text: str) -> int:
    TOKENS_PER_CHAR, HEADROOM, MIN_T, MAX_T = 0.86, 1.5, 192, 2048
    return max(MIN_T, min(MAX_T, int(len(text) * TOKENS_PER_CHAR * HEADROOM)))


# ---------------------------------------------------------------------------
# Same mixed-script EN/JP scenario as test_hotel_scenario_final.py's Run B
# (kept identical so results are directly comparable) — each chunk mixes
# English narration with a real-script Japanese clause in one sentence.
# ---------------------------------------------------------------------------
SCENARIO: list[tuple[str, str, str]] = [
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
    ("b04",
     "And if you're craving a snack, someone might say コンビニでしたら、歩いて二分ほどのところに"
     "ございますよ — the convenience store is just about a two-minute walk away. Isn't it "
     "wonderful how much warmth fits into one polite Japanese sentence?"),
    ("b05",
     "See how naturally that flowed — English explaining, then real Japanese right in the "
     "middle of the same breath? That's genuinely how bilingual conversation feels, not "
     "two separate languages bolted together!"),
]

_FOREIGN_SCRIPT_RANGES = [(0x3040, 0x9FFF), (0xAC00, 0xD7FF), (0x0400, 0x052F), (0x0600, 0x06FF)]


def has_japanese_script(text: str) -> bool:
    return any(0x3040 <= ord(ch) <= 0x9FFF for ch in text)


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
    base_model = Qwen3TTSModel.from_pretrained(
        base_path, device_map=device, dtype=dtype, attn_implementation=attn_impl,
    )
    log.info(f"Base model loaded in {time.time()-t0:.1f}s")
    return base_model, device, dtype, attn_impl


def generate_jp_anchor(device, dtype, attn_impl) -> str:
    """Loads VoiceDesign (separate model, unloaded after use — mirrors
    tts_service/server.py's own pattern of only loading VD when actually
    needed, not keeping two 1.7B models resident simultaneously), generates
    the Japanese-native Celeste-character anchor, saves it, and returns the
    saved path. Skips regeneration if already cached from a prior run."""
    anchor_path = os.path.join(OUT_DIR, "celeste_jp_anchor.wav")
    if os.path.exists(anchor_path):
        log.info(f"Using cached JP anchor: {anchor_path}")
        return anchor_path

    vd_path = resolve_model(VD_LOCAL, VD_HF_ID)
    log.info(f"Loading VoiceDesign model: {vd_path}...")
    t0 = time.time()
    vd_model = Qwen3TTSModel.from_pretrained(
        vd_path, device_map=device, dtype=dtype, attn_implementation=attn_impl,
    )
    log.info(f"VoiceDesign model loaded in {time.time()-t0:.1f}s")

    log.info("Generating Japanese-native Celeste-character anchor...")
    log.info(f"  instruct: {CELESTE_JP_INSTRUCT!r}")
    log.info(f"  text: {CELESTE_JP_ANCHOR_TEXT!r}")
    t0 = time.time()
    wavs, sr = vd_model.generate_voice_design(
        text=CELESTE_JP_ANCHOR_TEXT,
        language="Japanese",
        instruct=CELESTE_JP_INSTRUCT,
    )
    anchor_wav = wavs[0].astype(np.float32)
    log.info(f"  Generated {len(anchor_wav)/sr:.2f}s audio in {time.time()-t0:.1f}s @ {sr}Hz")
    sf.write(anchor_path, anchor_wav, sr)
    log.info(f"  Saved: {anchor_path}")

    del vd_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("VoiceDesign model unloaded.")
    return anchor_path


def build_clone_prompt(model, ref_wav_path: str, ref_text: str):
    ref_wav, ref_sr = sf.read(ref_wav_path)
    ref_wav = prepare_ref_audio(ref_wav.astype(np.float32), ref_sr)
    return model.create_voice_clone_prompt(
        ref_audio=(ref_wav, ref_sr), ref_text=ref_text, x_vector_only_mode=False,
    )


def run_single_reference(model, en_prompt) -> dict:
    """Baseline — identical to test_hotel_scenario_final.py's Run B: every
    chunk uses Celeste's English reference regardless of content language."""
    run_name = "run_single_ref_en"
    run_dir = os.path.join(OUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"=== RUN '{run_name}' (single English reference, baseline) ===")

    paths = []
    for chunk_id, text in SCENARIO:
        torch.manual_seed(GENERATION_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(GENERATION_SEED)
        wavs, sr = model.generate_voice_clone(
            text=text, language="auto", voice_clone_prompt=en_prompt,
            max_new_tokens=tokens_for_text(text), temperature=0.7, repetition_penalty=1.05,
        )
        clean_wav, leaked_s = strip_leading_reference_leak(wavs[0].astype(np.float32), sr)
        padded = pad_audio(clean_wav, sr)
        path = os.path.join(run_dir, f"{chunk_id}.wav")
        sf.write(path, padded, sr)
        paths.append(path)
        log.info(f"  {chunk_id}: {len(padded)/sr:.2f}s" + (f"  leak_stripped={leaked_s*1000:.0f}ms" if leaked_s else ""))

    _merge(paths, os.path.join(run_dir, f"_merged_{run_name}.wav"))
    return {"run_name": run_name, "paths": paths}


def run_dual_reference(model, en_prompt, jp_prompt) -> dict:
    """Per-chunk reference selection: chunks containing real Japanese script
    use the JP-native anchor's clone prompt; pure-English chunks (none in
    this scenario, but the logic is general) would use the English one.
    Note: every chunk in SCENARIO is itself English+Japanese MIXED within
    one sentence — this switches WHICH prompt conditions the WHOLE
    generate_voice_clone() call per chunk (still one call, one language
    value), not a mid-sentence prompt switch (the API has no such
    mechanism, per prior investigation). The bet being tested: does
    generating the ENTIRE mixed sentence against a Japanese-native
    reference produce better Japanese pronunciation than generating it
    against an English-native reference, even though both references are
    imperfect for the OTHER language present in the same sentence?"""
    run_name = "run_dual_ref"
    run_dir = os.path.join(OUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"=== RUN '{run_name}' (dual reference — JP anchor for JP-containing chunks) ===")

    paths = []
    for chunk_id, text in SCENARIO:
        use_jp = has_japanese_script(text)
        prompt = jp_prompt if use_jp else en_prompt
        lang = "auto"
        torch.manual_seed(GENERATION_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(GENERATION_SEED)
        wavs, sr = model.generate_voice_clone(
            text=text, language=lang, voice_clone_prompt=prompt,
            max_new_tokens=tokens_for_text(text), temperature=0.7, repetition_penalty=1.05,
        )
        clean_wav, leaked_s = strip_leading_reference_leak(wavs[0].astype(np.float32), sr)
        padded = pad_audio(clean_wav, sr)
        path = os.path.join(run_dir, f"{chunk_id}.wav")
        sf.write(path, padded, sr)
        paths.append(path)
        ref_used = "JP-anchor" if use_jp else "EN-anchor"
        log.info(f"  {chunk_id}  ref={ref_used}: {len(padded)/sr:.2f}s" +
                  (f"  leak_stripped={leaked_s*1000:.0f}ms" if leaked_s else ""))

    _merge(paths, os.path.join(run_dir, f"_merged_{run_name}.wav"))
    return {"run_name": run_name, "paths": paths}


def _merge(paths: list[str], merged_path: str) -> None:
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


def main():
    model, device, dtype, attn_impl = load_base_model()

    if not os.path.exists(CELESTE_REF_WAV):
        raise FileNotFoundError(f"Celeste reference clip not found: {CELESTE_REF_WAV}")

    jp_anchor_path = generate_jp_anchor(device, dtype, attn_impl)

    log.info("Building English clone prompt (celeste.wav, existing)...")
    en_prompt = build_clone_prompt(model, CELESTE_REF_WAV, CELESTE_REF_TEXT)

    log.info("Building Japanese clone prompt (newly generated anchor)...")
    jp_prompt = build_clone_prompt(model, jp_anchor_path, CELESTE_JP_ANCHOR_TEXT)

    run_single_reference(model, en_prompt)
    run_dual_reference(model, en_prompt, jp_prompt)

    log.info("=" * 90)
    log.info("COMPARISON — listen chunk-for-chunk (b00 vs b00, b01 vs b01, ...):")
    log.info(f"  Single English reference (today's approach):")
    log.info(f"    {os.path.join(OUT_DIR, 'run_single_ref_en', '_merged_run_single_ref_en.wav')}")
    log.info(f"  Dual reference (JP-native anchor for JP-containing chunks):")
    log.info(f"    {os.path.join(OUT_DIR, 'run_dual_ref', '_merged_run_dual_ref.wav')}")
    log.info(
        "\nWHAT TO LISTEN FOR:\n"
        "  1. Is the Japanese pronunciation in run_dual_ref noticeably more native-sounding\n"
        "     than in run_single_ref_en, for a fluent-Japanese listener specifically?\n"
        "  2. Does the JP anchor's VOICE CHARACTER still sound recognizably 'Celeste' —\n"
        "     gentle, calm, patient pace — or does it sound like an unrelated voice? (This\n"
        "     is the whole point of using VoiceDesign with a matching instruct string\n"
        "     instead of an off-the-shelf CustomVoice speaker like Ono_Anna.)\n"
        "  3. At the language-switch POINT within each mixed sentence (English handing off\n"
        "     to Japanese mid-sentence), does run_dual_ref sound like a natural code-switch\n"
        "     or like two different recordings spliced together? Both runs generate the\n"
        "     WHOLE mixed sentence in one generate_voice_clone() call against ONE reference\n"
        "     — dual_ref's reference is Japanese-native, so the ENGLISH portions of each\n"
        "     sentence might be the ones that suffer now (opposite direction from before) —\n"
        "     listen for that specifically.\n"
        "  4. If dual_ref's Japanese is much better but its English is now noticeably\n"
        "     worse, the real production fix is probably per-CLAUSE reference switching\n"
        "     (split each mixed sentence into an English sub-call + a Japanese sub-call,\n"
        "     each against its matching reference, then concatenate) rather than picking\n"
        "     one reference for the whole sentence — a further iteration, not implemented\n"
        "     here, but worth knowing as the next step if this run's tradeoff looks that way."
    )


if __name__ == "__main__":
    main()
