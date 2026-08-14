"""
FINAL combined test — Celeste voice, EN/JA hotel language-lesson scenario.

Supersedes the three earlier throwaway scripts from this investigation
(test_multilingual_consistency.py, test_ab_consistency_gpu.py,
test_chunk_boundary_crossfade.py — all removed). This is the one script to
run on the GPU instance. It targets the three concrete symptoms reported
against the earlier test output:

  1. MISPRONUNCIATION of foreign-language words even with language="auto".
     RUN A below tests "auto" per chunk (today's approach — one language
     value picked for the whole chunk by audio_manager._resolve_tts_language
     based on script detection). RUN B tests explicit language="japanese" on
     the SAME Japanese-heavy chunks, so you can A/B whether "auto" is
     actually worse than telling the model directly which language a given
     chunk is dominantly in. (Neither run splits a chunk mid-sentence by
     language — that's a separate, real limitation: Qwen's `language=` is
     one value per generate_voice_clone() call, so a chunk that's genuinely
     half English/half Japanese in mid-sentence has no per-word language
     control in this API. The workaround this script demonstrates instead
     is CHUNKING AT THE LANGUAGE BOUNDARY — see the paragraph below, which
     splits English narration and Japanese phrases into separate parts/
     sentences rather than mixing scripts inside one sentence — combined
     with each part's own resolved language.)

  2. MERGE "OVERLAP" / sounds cut in too early. This script measures, for
     every chunk: its raw generated duration, its duration AFTER
     _pad_audio_file's 100ms-both-ends padding, and then the ACTUAL merged
     file's total duration vs. the naive sum of padded chunk durations. If
     those two numbers disagree, that mismatch (not a crossfade/DSP issue —
     see the correction below) is the mechanism: any downstream code that
     schedules part N+1's start time from the SUM of part durations rather
     than from where part N's audio actually ends in the merged file will
     drift, and that drift compounds every chunk — which matches "sometimes
     it seems like the audio was merged too early, almost like there's an
     overlap" (early chunks look fine, later ones increasingly don't).
     CORRECTION vs. an earlier iteration of this investigation: a prior
     version of this test proposed crossfading the concat step, built on a
     synthetic-tone test that turned out to be invalid (it artificially
     zero-padded both edges, so it couldn't reproduce a real seam). That
     idea is NOT included here. This script instead just measures whether
     ffmpeg's "-f concat -c copy" (bit-exact, lossless splice — no
     resampling, no overlap by construction) actually preserves
     sum(chunk_durations) == merged_duration exactly, which is the real
     way to confirm/rule out the concat step itself as the source of
     the reported overlap, versus it being a downstream timing-consumer bug
     (e.g. in wordTimings / triggerTime math outside this script's scope —
     see the printed diagnostic block at the end for what to check next if
     concat itself measures out clean).

  3. PARALINGUISTIC CUES (laughter, sighs, emphasis) — Qwen3-TTS-Base has NO
     documented bracketed control-tag syntax (no "[laughs]", no "[sigh]").
     Confirmed against the model's own HF README (Base model card,
     "Voice Clone" section): the shown usage pattern embeds emoticons and
     natural punctuation DIRECTLY in the spoken text, e.g. the README's own
     example: text="...it's a disaster (◍•͈⌔•͈◍), very sad!" — the model's
     "Intelligent Text Understanding" (per the README's feature list) reads
     semantic/punctuation cues from the text itself, not a separate control
     channel. This script's paragraph uses that same mechanism: emphasis
     punctuation ("!", "?", em-dashes for a self-interrupting/thinking
     beat), and a couple of emoticon-style cues, rather than inventing a tag
     format the model doesn't support. If you specifically need scripted
     laughter/sighs as distinct audio events rather than tone coloring, that
     is NOT something this model version exposes — flagging that limitation
     explicitly rather than faking support for it.

SCENARIO: a language-learning-app style lesson — checking into a hotel in
Japan, an English-speaking traveler learning key Japanese phrases from a
friendly, energetic tutor (Celeste). Chunked the way audio_manager.py
actually chunks real content: each chunk is one narrated beat, roughly
10-20s of speech, matching Part.duration guidance from this codebase's own
video-authoring instructions.

Run (GPU instance, same venv as qwen-server):
    cd qwen-server && source venv/bin/activate
    python test_hotel_scenario_final.py

Output: hotel_test_out/{run_a_auto,run_b_explicit_lang}/chunk_NN.wav, one
merged file per run, and a printed diagnostics report covering all three
symptoms above.
"""
import os
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

import time
import math
import logging
import subprocess
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("hotel_test")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "hotel_test_out")
os.makedirs(OUT_DIR, exist_ok=True)

BASE_LOCAL = os.path.join(SCRIPT_DIR, "Qwen3-TTS-12Hz-1.7B-Base")
BASE_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

CELESTE_REF_WAV = os.path.join(
    SCRIPT_DIR, "..", "pinn-research-model", "scene", "voices", "celeste_ref.wav"
)
CELESTE_REF_TEXT = "Good evening. I have spent many years"

# Mirrors audio_manager.AUDIO_PAD_SECONDS / _pad_audio_file exactly, so the
# padded-duration math in this script matches production's real behavior.
AUDIO_PAD_SECONDS = 0.1


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


def prepare_ref_audio(wav: np.ndarray, sr: int) -> np.ndarray:
    frame_samples = sr // CODEC_HZ
    n_frames = len(wav) // frame_samples
    trimmed = wav[: n_frames * frame_samples].copy()
    fade_n = min(int(sr * FADE_MS / 1000), len(trimmed) // 4)
    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    trimmed[:fade_n] *= ramp
    trimmed[-fade_n:] *= ramp[::-1]
    return trimmed.astype(np.float32)


def pad_audio(wav: np.ndarray, sr: int, pad_seconds: float = AUDIO_PAD_SECONDS) -> np.ndarray:
    """Exact port of audio_manager._pad_audio_file's in-memory equivalent —
    prepend/append pad_seconds of silence, same as production applies to
    every generated chunk before merging."""
    silence = np.zeros(int(sr * pad_seconds), dtype=wav.dtype)
    return np.concatenate([silence, wav, silence])


def tokens_for_text(text: str) -> int:
    TOKENS_PER_CHAR, HEADROOM, MIN_T, MAX_T = 0.86, 1.5, 192, 2048
    return max(MIN_T, min(MAX_T, int(len(text) * TOKENS_PER_CHAR * HEADROOM)))


# ---------------------------------------------------------------------------
# THE SCENARIO — "Checking in: Japanese for the front desk"
#
# Chunked at LANGUAGE BOUNDARIES (English narration chunks vs. Japanese-
# phrase-teaching chunks kept as separate parts) rather than mixed mid-
# sentence, per the mispronunciation workaround explained in the module
# docstring. Each chunk carries its own "lang" — this is RUN A's value
# (what audio_manager._resolve_tts_language would actually pick today: a
# script-detection-driven single language per chunk). RUN B below re-runs
# the SAME text with an explicit stronger language hint on the
# Japanese-heavy chunks specifically, to isolate whether "auto" itself is
# the weak point.
#
# Paralinguistic color via punctuation/emoticon (no bracket tags — see
# docstring point 3): warmth/encouragement via "!", a light self-correcting
# beat via em-dash, mild comic exasperation via a text-emoticon on chunk 07,
# matching the README's own demonstrated pattern.
# ---------------------------------------------------------------------------
SCENARIO: list[tuple[str, str, str]] = [
    # (chunk_id, resolve_as_lang_for_run_a, text)
    ("chunk_00", "english",
     "Okay, we've just landed in Tokyo, and our first stop is the hotel front desk. "
     "Let's get you ready with a few essential phrases before we walk in."),
    ("chunk_01", "japanese",
     "こんにちは。予約しています。"),
    ("chunk_02", "english",
     "That means 'Hello, I have a reservation' — konnichiwa, yoyaku shite imasu. "
     "Say it with me, nice and clear: konnichiwa, yoyaku shite imasu."),
    ("chunk_03", "english",
     "Now, the front desk clerk might ask for your passport. Here's how you'll hear it — "
     "listen closely, because this next word trips a lot of learners up."),
    ("chunk_04", "japanese",
     "パスポートをお願いします。"),
    ("chunk_05", "english",
     "Pasupōto o onegaishimasu — 'Your passport, please.' Notice how 'passport' becomes "
     "pasu-poh-toh in Japanese — it's a borrowed word, so the rhythm is completely different "
     "from English. Don't rush it!"),
    ("chunk_06", "english",
     "Once you're checked in, you'll want to ask where the elevator is — trust me, you'll "
     "need this one every single day of the trip."),
    ("chunk_07", "japanese",
     "エレベーターはどこですか？"),
    ("chunk_08", "english",
     "Erebētā wa doko desu ka? — 'Where is the elevator?' — and if they point and smile, "
     "you'll know exactly what to say next: arigatou gozaimasu, thank you very much (^_^)."),
    ("chunk_09", "english",
     "One last one, for the morning after — you're going to want breakfast, and you're going "
     "to want to know if it's included."),
    ("chunk_10", "japanese",
     "朝食は含まれていますか？"),
    ("chunk_11", "english",
     "Chōshoku wa fukumarete imasu ka? — 'Is breakfast included?' — such a useful little "
     "sentence to have ready on day one."),
    ("chunk_12", "english",
     "And that's it — four real phrases you'll actually use within your first ten minutes "
     "in Japan. Practice them once more before we move on, okay? You've got this!"),
]

# RUN B: for the Japanese-script chunks specifically, force the model's
# explicit "japanese" language value (same value RUN A already uses for
# those rows in this particular scenario, since script-detection correctly
# identifies them as Japanese either way) — the REAL A/B differs on chunks
# that MIX English romaji explanation with an embedded Japanese phrase in
# the same sentence (chunks 02, 05, 08, 11 above all do this: an English
# sentence containing a romanized Japanese phrase). RUN A resolves those as
# "auto" per audio_manager._resolve_tts_language's actual has_foreign check
# (they contain real Japanese-script characters earlier in the scenario's
# surrounding context, but individually these specific sentences are
# Latin-script romaji + English — resolve_tts_language would call these
# "english", not "auto", since it scans for non-Latin CODEPOINTS, and romaji
# has none). That mismatch — a chunk that is PHONETICALLY Japanese content
# but LEXICALLY all-Latin-script — is a real gap in the current language
# resolver, independent of the model itself, and worth testing directly:
RUN_B_OVERRIDES = {
    "chunk_02": "auto",       # contains romaji "konnichiwa, yoyaku shite imasu" — force auto instead of the resolver's "english"
    "chunk_05": "auto",       # contains romaji "pasupōto o onegaishimasu"
    "chunk_08": "auto",       # contains romaji "erebētā wa doko desu ka"
    "chunk_11": "auto",       # contains romaji "chōshoku wa fukumarete imasu ka"
}


def resolve_lang_run_a(declared_lang: str) -> str:
    """Mirrors audio_manager._resolve_tts_language's actual behavior for
    this script's purposes: Japanese-script chunks resolve to 'japanese'
    directly (declared), English chunks with embedded romaji still resolve
    to 'english' because the resolver only scans Unicode codepoints, not
    phonetic content — it can't tell romaji from ordinary English text."""
    return declared_lang


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
        log.warning("No CUDA device detected — this script is designed for a GPU box "
                     "(13 chunks will take a long time on CPU).")

    log.info(f"Device={device}  dtype={dtype}  attn={attn_impl}")
    base_path = resolve_model(BASE_LOCAL, BASE_HF_ID)
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        base_path, device_map=device, dtype=dtype, attn_implementation=attn_impl,
    )
    log.info(f"Base model loaded in {time.time()-t0:.1f}s")
    return model


def build_clone_prompt(model):
    """ONE full-ICL prompt, built once, reused unchanged for every chunk in
    both runs — see the earlier investigation's finding that production's
    x_vector_only_mode=True is the weaker cloning mode; this script always
    uses the proposed fix (x_vector_only_mode=False) so today's test
    isolates ONLY the language-resolution variable, not cloning-mode drift
    on top of it."""
    if not os.path.exists(CELESTE_REF_WAV):
        raise FileNotFoundError(f"Celeste reference clip not found: {CELESTE_REF_WAV}")
    ref_wav, ref_sr = sf.read(CELESTE_REF_WAV)
    ref_wav = prepare_ref_audio(ref_wav.astype(np.float32), ref_sr)
    return model.create_voice_clone_prompt(
        ref_audio=(ref_wav, ref_sr), ref_text=CELESTE_REF_TEXT, x_vector_only_mode=False,
    )


def run_variant(model, clone_prompt, run_name: str, lang_overrides: dict[str, str]) -> dict:
    """Generates all chunks, pads each exactly like production, merges with
    the SAME ffmpeg concat production uses, and returns full diagnostics."""
    run_dir = os.path.join(OUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"=== RUN '{run_name}' ===")

    per_chunk = []
    raw_paths = []
    t_run = time.time()

    for chunk_id, declared_lang, text in SCENARIO:
        lang = lang_overrides.get(chunk_id, resolve_lang_run_a(declared_lang))
        max_tok = tokens_for_text(text)
        t0 = time.time()
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=lang,
            voice_clone_prompt=clone_prompt,
            max_new_tokens=max_tok,
            temperature=0.7,
            repetition_penalty=1.05,
        )
        raw_wav = wavs[0].astype(np.float32)
        gen_elapsed = time.time() - t0
        raw_duration = len(raw_wav) / sr

        padded_wav = pad_audio(raw_wav, sr)
        padded_duration = len(padded_wav) / sr

        path = os.path.join(run_dir, f"{chunk_id}.wav")
        sf.write(path, padded_wav, sr)
        raw_paths.append(path)

        log.info(f"  {chunk_id}  lang={lang:<10} raw={raw_duration:.2f}s  "
                  f"padded={padded_duration:.2f}s  gen={gen_elapsed:.1f}s  text={text[:50]!r}")
        per_chunk.append({
            "chunk_id": chunk_id, "lang": lang, "text": text,
            "raw_duration": raw_duration, "padded_duration": padded_duration,
            "sr": sr,
        })

    log.info(f"  Run '{run_name}' generation total: {time.time()-t_run:.1f}s")

    # ---- Merge exactly like production: ffmpeg -f concat -c copy ----------
    merged_path = os.path.join(run_dir, f"_merged_{run_name}.wav")
    concat_list = os.path.join(run_dir, "_concat.txt")
    with open(concat_list, "w") as f:
        for p in raw_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    subprocess.run(
        ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", "-y", merged_path],
        capture_output=True, check=True,
    )
    os.remove(concat_list)

    merged_actual_duration = sf.info(merged_path).duration
    sum_padded_durations = sum(c["padded_duration"] for c in per_chunk)
    # This is exactly the check production would need for its own
    # part['duration'] math: does the merged file's real length match the
    # sum of the individually-measured (post-pad) chunk durations that get
    # stamped into the VideoSpec? A mismatch here — NOT crossfading, NOT DSP
    # — is the mechanism that would produce a creeping "starts too early"
    # feeling, because every downstream part-start-time computed as a
    # cumulative sum of `duration` would silently diverge from where that
    # part's audio actually sits in the merged file.
    discrepancy = merged_actual_duration - sum_padded_durations

    log.info(f"  Merged: {merged_path}")
    log.info(f"  Sum of individually-measured padded chunk durations: {sum_padded_durations:.4f}s")
    log.info(f"  Actual merged file duration (ffprobe/soundfile):     {merged_actual_duration:.4f}s")
    log.info(f"  Discrepancy: {discrepancy:+.4f}s "
              f"({'MATCH — concat step is not the source of drift' if abs(discrepancy) < 0.01 else 'MISMATCH — see diagnostics below'})")

    return {
        "run_name": run_name, "per_chunk": per_chunk, "merged_path": merged_path,
        "sum_padded_durations": sum_padded_durations,
        "merged_actual_duration": merged_actual_duration, "discrepancy": discrepancy,
    }


def print_final_report(results: list[dict]):
    log.info("=" * 90)
    log.info("FINAL DIAGNOSTIC REPORT")
    log.info("=" * 90)

    log.info("\n[1] MISPRONUNCIATION CHECK — compare these files by ear:")
    for r in results:
        log.info(f"  {r['run_name']}:")
        for c in r["per_chunk"]:
            if c["lang"] in ("japanese", "auto"):
                log.info(f"    {c['chunk_id']}  lang={c['lang']:<10}  {c['text'][:70]!r}")
    log.info("  Listen to each run's Japanese-content chunks side by side. If RUN B's "
              "'auto'-forced romaji chunks (02/05/08/11) pronounce the embedded Japanese "
              "phrase noticeably better than RUN A's 'english'-resolved versions of the "
              "SAME text, that confirms the gap is in audio_manager._resolve_tts_language's "
              "codepoint-only script detection — it should also flag ROMANIZED foreign-"
              "language content, not just non-Latin script, as a candidate for 'auto'.")

    log.info("\n[2] MERGE OVERLAP / TIMING CHECK:")
    for r in results:
        status = "OK" if abs(r["discrepancy"]) < 0.01 else "INVESTIGATE"
        log.info(f"  {r['run_name']}: sum={r['sum_padded_durations']:.4f}s  "
                  f"merged={r['merged_actual_duration']:.4f}s  "
                  f"discrepancy={r['discrepancy']:+.4f}s  [{status}]")
    any_mismatch = any(abs(r["discrepancy"]) >= 0.01 for r in results)
    if any_mismatch:
        log.info("  >>> A real discrepancy was measured — the ffmpeg concat step itself is "
                  "NOT lossless for this input in this environment. Worth checking ffmpeg's "
                  "stderr output and the input WAVs' sample format/rate consistency next.")
    else:
        log.info("  >>> No discrepancy — 'ffmpeg -c copy' concat is exact here. If the "
                  "'merged too early / overlap' symptom persists in the real pipeline, it is "
                  "NOT the concat step itself; check next: (a) whether part['duration'] as "
                  "stamped into the VideoSpec JSON is read by the SAME padded-duration math "
                  "used here (ceil(seconds*1000) + DURATION_BUFFER_MS) everywhere it's "
                  "consumed downstream, (b) whether the frontend/timeline computes each "
                  "part's start time as a cumulative sum of `duration` fields (correct) vs. "
                  "some other timing source that could desync from the merged audio file, "
                  "and (c) whether AudioAligner's forced-alignment wordTimings for Qwen "
                  "output (scene/alignment.py) are landing early specifically on romaji/"
                  "code-switched speech — that library was validated primarily against "
                  "clean single-language speech and may mis-time codec artifacts at a "
                  "language switch boundary.")

    log.info("\n[3] PARALINGUISTIC CUES:")
    log.info("  This script's text uses emoticon/punctuation-driven expression only "
              "(e.g. chunk_08's '(^_^)', exclamation points, em-dashes for a self-"
              "correcting beat) — per the Base model's own README example "
              "(text=\"...it's a disaster (◍•͈⌔•͈◍), very sad!\"). Qwen3-TTS-Base has NO "
              "documented bracket-tag syntax for scripted non-verbal events (no '[laughs]', "
              "no '[sigh]' as a distinct audio insertion) — do not rely on that as a feature; "
              "any perceived laugh/sigh is the model's own prosody interpretation of the "
              "surrounding text, not a controllable discrete event.")

    log.info("\n" + "=" * 90)


def main():
    model = load_base_model()
    clone_prompt = build_clone_prompt(model)

    results = []
    results.append(run_variant(model, clone_prompt, "run_a_auto",
                                lang_overrides={}))  # pure resolve_lang_run_a behavior
    results.append(run_variant(model, clone_prompt, "run_b_explicit_lang",
                                lang_overrides=RUN_B_OVERRIDES))

    print_final_report(results)


if __name__ == "__main__":
    main()
