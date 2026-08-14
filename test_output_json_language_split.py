"""
A/B test reproducing pinnlab-animate/examples/output.json's exact reported
failure mode, then testing the fix — driven directly from the real JSON
files rather than hand-copied strings, so this test stays correct if either
file changes later.

CONTEXT: output.json (a Japanese-language-learning video) was narrated
through generate_audio_for_transcript() with voice_model="normal" (Kokoro,
English-only, no language routing at all) instead of voice_model="emotion"
(Qwen, the multilingual path with per-part language resolution). On top of
that, every part's transcript mixed real Japanese script, a parenthetical
ROMANIZATION of the same phrase, and English narration in ONE string — e.g.
the original (pre-fix) scene1/sec0/part0:

    "Every conversation starts with a greeting, and the most common one in
    Japanese is こんにちは (konnichiwa). It simply means hello, and you can
    use it all through the daytime with anyone you meet."

Reported symptoms: audio that goes "scratchy," "like someone is drowning";
romanized Japanese read with English phonetics inside the same audio chunk;
and the same Japanese phrase spoken twice — once correctly (real script),
once mispronounced (immediately after, as its own romanization).

ROOT CAUSE (two independent bugs, both upstream of the TTS/service layer,
which is already correct — see scene/audio_manager.py's
_resolve_scoped_language/_resolve_tts_language, scene/voice_language_profiles.py,
and tts_service/server.py's per-(voice,language) warmed clone prompts):

  1. The Kokoro path has NO language routing at all —
     _resolve_tts_language/_resolve_scoped_language only run on the Qwen
     ("emotion") branch. Kokoro's English model reads raw kanji/hiragana
     bytes as garbage phonemes — this alone explains the "drowning" artifact.
     FIXED: multilingual content is now Pro/Enterprise-tier only, ENFORCED
     at runtime in scene/audio_manager.py's AudioManager.__init__ (downgrades
     voice_model="emotion" -> "normal" whenever subscription_type isn't
     pro/enterprise) — not just requested via a prompt tier flag.

  2. The transcript ITSELF was malformed: it mixed real-script Japanese with
     an inline English-phonetic romanization in ONE string. lib/main.py's
     prompt used to literally instruct the model to do this ("Add phonetic
     hints after non-Latin words... Japanese: さようなら (sa-yo-u-na-ra)"),
     directly contradicting its own hard rule at instructions.txt Part 5A.
     FIXED: lib/main.py's prompt no longer contains this contradiction (all
     few-shot examples now split mixed-language content into separate
     single-language parts, matching Part 5A), and output.json itself has
     been rewritten the same way (see
     qwen-server/output_json_original_repro_snapshot.json for the pre-fix
     version this test's Run A reproduces against).

THIS SCRIPT tests three variants side by side, using REAL parts loaded
directly from JSON (not hand-copied) — scene 1, "Common Phrases for
Beginners", the most mixed-script-dense section:

  RUN A (repro)   — parts loaded from output_json_original_repro_snapshot.json
                     (the pre-fix file, frozen as a permanent regression
                     fixture): real Japanese + parenthetical romanization +
                     English, one chunk, one generate_voice_clone() call,
                     language="auto" (mirrors _resolve_tts_language's actual
                     behavior on script-mixed text with no explicit
                     language field, which is what that file had).

  RUN B (fix)     — parts loaded LIVE from
                     pinnlab-animate/examples/output.json (the corrected
                     file): each Japanese phrase is already split into its
                     own part with language="japanese", no romanization
                     anywhere, exactly how
                     scene/audio_manager.py._resolve_scoped_language +
                     part.language overrides are designed to be driven.

  RUN C (fix, dual-reference) — same parts as Run B, but Japanese parts use
                     a VoiceDesign-generated Japanese-native reference for
                     the SAME voice character (mirrors
                     test_dual_reference_jp.py's approach) instead of the
                     English-only reference. Tests whether dual-reference
                     measurably improves the Japanese parts' naturalness
                     over single-reference + language="japanese".

Run (GPU instance recommended — CPU works but is slow):
    cd qwen-server && source venv/bin/activate
    python test_output_json_language_split.py

Output: output_json_lang_test_out/{run_a_repro, run_b_split_fix,
run_c_split_dual_ref}/*.wav, plus a diagnostic report. Listen to run_a vs.
run_b/run_c back to back on the SAME teaching beats to A/B the fix.
"""
import os
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

import json
import re
import time
import logging
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("output_json_lang_test")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output_json_lang_test_out")
os.makedirs(OUT_DIR, exist_ok=True)

BASE_LOCAL = os.path.join(SCRIPT_DIR, "Qwen3-TTS-12Hz-1.7B-Base")
BASE_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VD_LOCAL = os.path.join(SCRIPT_DIR, "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
VD_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# Frozen pre-fix snapshot (copy of output.json as it stood before the
# language-split fix) — a permanent regression fixture, independent of
# whatever output.json contains now.
REPRO_SNAPSHOT_PATH = os.path.join(SCRIPT_DIR, "output_json_original_repro_snapshot.json")
# The live, corrected file — loaded fresh on every run so this test tracks
# whatever output.json actually contains, not a stale copy. qwen-server/ and
# pinnlab-animate/ are sibling directories (see NOVA_REF_WAV below for the
# same sibling-path pattern against pinn-research-model/).
LIVE_OUTPUT_JSON_PATH = os.path.join(SCRIPT_DIR, "..", "pinnlab-animate", "examples", "output.json")

# Nova is output.json's voice (voice_settings.voice) — this test uses the
# same voice/character for all runs.
NOVA_REF_WAV = os.path.join(SCRIPT_DIR, "..", "pinn-research-model", "scene", "voices", "nova_ref.wav")
NOVA_REF_TEXT = "Well done. You've covered the core idea."
NOVA_JP_INSTRUCT = (
    "温かく表現力豊かな日本語の女性の声。自信を持った、ほどよい速さで、"
    "はっきりとした発音。親しみやすく、まるで大好きな話題を語る博識な友人のような口調。"
)
NOVA_JP_ANCHOR_TEXT = "こんにちは、ようこそ。一緒に面白いことを探っていきましょう。"

AUDIO_PAD_SECONDS = 0.05  # matches scene/audio_manager.AUDIO_PAD_SECONDS
REF_TRAILING_SILENCE_S = 0.5
GENERATION_SEED = 42

CODEC_HZ = 12
FADE_MS = 30

# scene1 = "Common Phrases for Beginners" — index 1 in output.json's scenes[].
TARGET_SCENE_INDEX = 1
# Cap on how many original (pre-split) teaching beats to pull, so a CPU dev
# run stays short — None = all of them.
MAX_BEATS = 5


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
    """Codec-frame-align + edge-fade + trailing silence — exact port of
    scene/qwen_voice.py's prepare_ref_audio (see that module for the full
    rationale on why each step exists)."""
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
    """Exact port of audio_manager._pad_audio_file's in-memory equivalent."""
    silence = np.zeros(int(sr * pad_seconds), dtype=wav.dtype)
    return np.concatenate([silence, wav, silence])


def tokens_for_text(text: str) -> int:
    """Exact port of scene/qwen_voice.py's tokens_for_text."""
    TOKENS_PER_CHAR, HEADROOM, MIN_T, MAX_T = 0.86, 1.5, 192, 2048
    return max(MIN_T, min(MAX_T, int(len(text) * TOKENS_PER_CHAR * HEADROOM)))


_FOREIGN_SCRIPT_RANGES = [(0x3040, 0x9FFF), (0xAC00, 0xD7FF), (0x0400, 0x052F), (0x0600, 0x06FF)]


def has_foreign_script(text: str) -> bool:
    return any(lo <= ord(ch) <= hi for ch in text for lo, hi in _FOREIGN_SCRIPT_RANGES)


def load_repro_scenario(max_beats: int | None = MAX_BEATS) -> list[tuple[str, str, str]]:
    """Loads REAL, pre-fix parts straight from the frozen snapshot — scene 1's
    sections/parts, each part's raw transcript, tagged language="auto" (what
    _resolve_tts_language actually resolves script-mixed no-language-field
    text to)."""
    with open(REPRO_SNAPSHOT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    scene = data["scenes"][TARGET_SCENE_INDEX]
    scenario = []
    for section in scene.get("sections", []):
        for part in section.get("parts", []):
            text = part.get("transcript", "").strip()
            if not text:
                continue
            lang = "auto" if has_foreign_script(text) else "english"
            scenario.append((part["id"], lang, text))
            if max_beats is not None and len(scenario) >= max_beats:
                return scenario
    return scenario


_SPLIT_SUFFIX_RE = re.compile(r"-(en|jp)\d+$")


def load_fixed_scenario(max_beats: int | None = MAX_BEATS) -> list[tuple[str, str, str]]:
    """Loads REAL, post-fix parts straight from the LIVE output.json — already
    split at the language boundary, each part tagged with its own explicit
    `language` field (or inheriting the video-level "english" default when
    absent, e.g. beats with no foreign content). Caps to roughly the same
    number of ORIGINAL teaching beats as load_repro_scenario for a fair
    side-by-side (each original beat is now 1-5 split parts, so this walks
    original-beat boundaries via each part id's base name — the id with any
    "-en<N>"/"-jp<N>" split suffix stripped off — not a flat part count)."""
    with open(LIVE_OUTPUT_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    scene = data["scenes"][TARGET_SCENE_INDEX]
    scenario = []
    seen_base_ids: set[str] = set()
    for section in scene.get("sections", []):
        for part in section.get("parts", []):
            text = part.get("transcript", "").strip()
            if not text:
                continue
            base_id = _SPLIT_SUFFIX_RE.sub("", part["id"])
            if max_beats is not None and base_id not in seen_base_ids and len(seen_base_ids) >= max_beats:
                return scenario
            seen_base_ids.add(base_id)
            lang = part.get("language", "english")
            scenario.append((part["id"], lang, text))
    return scenario


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
        log.warning("No CUDA device detected — this will be slow on CPU but will still run "
                     "(useful for a quick structural smoke test of the split logic; run on "
                     "GPU for a real listening test).")
    log.info(f"Device={device}  dtype={dtype}  attn={attn_impl}")

    base_path = resolve_model(BASE_LOCAL, BASE_HF_ID)
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        base_path, device_map=device, dtype=dtype, attn_implementation=attn_impl,
    )
    log.info(f"Base model loaded in {time.time()-t0:.1f}s")
    return model, device, dtype, attn_impl


def generate_jp_anchor(device, dtype, attn_impl) -> str:
    """VoiceDesign-generated Japanese-native anchor for Nova's character —
    same approach as test_dual_reference_jp.py's celeste_jp_anchor. Cached
    to disk after first run."""
    anchor_path = os.path.join(OUT_DIR, "nova_jp_anchor.wav")
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

    log.info("Generating Japanese-native Nova-character anchor...")
    t0 = time.time()
    wavs, sr = vd_model.generate_voice_design(
        text=NOVA_JP_ANCHOR_TEXT, language="Japanese", instruct=NOVA_JP_INSTRUCT,
    )
    anchor_wav = wavs[0].astype(np.float32)
    log.info(f"  Generated {len(anchor_wav)/sr:.2f}s audio in {time.time()-t0:.1f}s @ {sr}Hz")
    sf.write(anchor_path, anchor_wav, sr)

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


def run_variant(
    model, run_name: str, scenario: list[tuple[str, str, str]],
    en_prompt, jp_prompt=None,
) -> dict:
    """Generates every part in `scenario`. If jp_prompt is given, parts
    tagged language="japanese" use it (dual-reference); everything else
    (including "auto"/"english") uses en_prompt (single-reference,
    matching production's current single-clone-prompt-per-voice design)."""
    run_dir = os.path.join(OUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"=== RUN '{run_name}' ({len(scenario)} part(s)) ===")

    paths, durations = [], []
    t_run0 = time.time()
    for part_id, lang, text in scenario:
        prompt = jp_prompt if (jp_prompt is not None and lang == "japanese") else en_prompt
        torch.manual_seed(GENERATION_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(GENERATION_SEED)
        t0 = time.time()
        wavs, sr = model.generate_voice_clone(
            text=text, language=lang, voice_clone_prompt=prompt,
            max_new_tokens=tokens_for_text(text), temperature=0.7, repetition_penalty=1.05,
        )
        elapsed = time.time() - t0
        padded = pad_audio(wavs[0].astype(np.float32), sr)
        safe_id = part_id.replace("/", "_")
        path = os.path.join(run_dir, f"{safe_id}.wav")
        sf.write(path, padded, sr)
        dur = len(padded) / sr
        paths.append(path)
        durations.append(dur)
        log.info(f"  [{part_id}] lang={lang:8s} {dur:5.2f}s audio in {elapsed:5.2f}s  ->  {path}")

    total_elapsed = time.time() - t_run0
    log.info(f"=== '{run_name}' done: {len(scenario)} part(s), "
              f"{sum(durations):.1f}s total audio, {total_elapsed:.1f}s wall time ===")
    return {"run_name": run_name, "paths": paths, "durations": durations, "elapsed": total_elapsed}


def print_final_report(results: list[dict], repro_scenario, fixed_scenario):
    print("\n" + "=" * 78)
    print("REPORT — output.json language-mixing repro vs. split-fix")
    print("=" * 78)
    print(f"\nRun A source: {REPRO_SNAPSHOT_PATH}  ({len(repro_scenario)} part(s), pre-fix)")
    print(f"Run B/C source: {LIVE_OUTPUT_JSON_PATH}  ({len(fixed_scenario)} part(s), live/current)")
    for r in results:
        print(f"\n{r['run_name']}:")
        print(f"  parts: {len(r['paths'])}   total audio: {sum(r['durations']):.1f}s   "
              f"wall time: {r['elapsed']:.1f}s")
        print(f"  dir: {os.path.dirname(r['paths'][0]) if r['paths'] else '(none)'}")
    print("\nHOW TO EVALUATE:")
    print("  1. Listen to run_a_repro/*.wav — this is what output.json's Qwen path would have")
    print("     produced on the ORIGINAL, pre-fix file (production actually used Kokoro, which")
    print("     is worse: no language routing at all). Listen for garbled/scratchy transitions")
    print("     at each parenthetical romanization, and words effectively said twice.")
    print("  2. Listen to run_b_split_fix/*.wav — should be clean native Japanese on the")
    print("     '-jp*' parts, natural English on the '-en*' parts, no romanization, no")
    print("     repetition. These are loaded LIVE from output.json, so they reflect whatever")
    print("     that file currently contains.")
    print("  3. Listen to run_c_split_dual_ref/*_jp*.wav vs run_b's — same parts, Japanese")
    print("     parts use a JP-native reference instead of Nova's English one. If run_c sounds")
    print("     more natural, that's the CLONE_MODE_X_VECTOR_ONLY / dual-reference")
    print("     investigation (tts_service/server.py line ~101) validated against THIS")
    print("     content, not just the hotel scenario.")
    print("=" * 78 + "\n")


def main():
    if not os.path.exists(NOVA_REF_WAV):
        raise FileNotFoundError(
            f"Nova reference clip not found: {NOVA_REF_WAV}\n"
            "Run from qwen-server/ inside a checkout that has pinn-research-model/ as a sibling."
        )
    if not os.path.exists(REPRO_SNAPSHOT_PATH):
        raise FileNotFoundError(
            f"Repro snapshot not found: {REPRO_SNAPSHOT_PATH}\n"
            "This should be a permanent, committed fixture (copy of output.json as it stood "
            "before the language-split fix) — see the module docstring."
        )
    if not os.path.exists(LIVE_OUTPUT_JSON_PATH):
        raise FileNotFoundError(f"Live output.json not found: {LIVE_OUTPUT_JSON_PATH}")

    repro_scenario = load_repro_scenario()
    fixed_scenario = load_fixed_scenario()
    log.info(f"Loaded {len(repro_scenario)} pre-fix part(s) from snapshot, "
              f"{len(fixed_scenario)} post-fix part(s) from live output.json")

    model, device, dtype, attn_impl = load_base_model()
    en_prompt = build_clone_prompt(model, NOVA_REF_WAV, NOVA_REF_TEXT)
    jp_anchor_path = generate_jp_anchor(device, dtype, attn_impl)
    jp_prompt = build_clone_prompt(model, jp_anchor_path, NOVA_JP_ANCHOR_TEXT)

    results = []
    results.append(run_variant(model, "run_a_repro", repro_scenario, en_prompt))
    results.append(run_variant(model, "run_b_split_fix", fixed_scenario, en_prompt))
    results.append(run_variant(model, "run_c_split_dual_ref", fixed_scenario, en_prompt, jp_prompt=jp_prompt))

    print_final_report(results, repro_scenario, fixed_scenario)


if __name__ == "__main__":
    main()
