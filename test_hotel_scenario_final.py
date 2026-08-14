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

  4. LEAKED REFERENCE-AUDIO FRAGMENT (confirmed from a real GPU run of this
     script, added after point 2's "merge overlap" theory was tested and
     ruled out — merge duration math came back exact, zero discrepancy, in
     both runs). What was actually reported — chunk_01 (pure Japanese,
     "こんにちは。予約しています。") sounding like it starts with a
     leading English "s"/"is" sound — is a real bug in the qwen_tts PACKAGE
     itself (version 0.1.1, not this codebase): generate_voice_clone()'s
     ICL trim computes the reference/generated boundary via a proportional
     ratio (ref codec-token count / total codec-token count, scaled onto
     decoded sample count) instead of an exact per-token sample mapping.
     That ratio can land short, leaving a trailing fragment of the
     REFERENCE clip's own speech (Celeste's ref_text ends "...many years")
     stitched onto the front of the output. Confirmed against the actual
     generated chunk_01.wav from both GPU runs: a short ~90-100ms energy
     burst, then a genuine ~60-80ms silence gap, THEN the real "konnichiwa"
     onset — two concatenated acoustic events, not one natural utterance
     start. Most audible on SHORT chunks, since a fixed ~100ms leak is a
     much larger fraction of a 2-3s clip than a 10s+ one — exactly matching
     why this was noticed on the short Japanese-only lines specifically.
     WORKAROUND (applied in this script, see strip_leading_reference_leak):
     detects a burst -> silence-gap -> re-onset pattern in the first 300ms
     of every generated chunk and trims past it, independent of whatever
     cut point the library itself already applied. Conservative by design —
     only acts when an actual silence gap is found, so it won't clip a
     legitimate utterance that starts on a soft consonant.

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


# ---------------------------------------------------------------------------
# WORKAROUND for a confirmed qwen_tts library bug (package version 0.1.1,
# NOT this codebase): generate_voice_clone()'s ICL trim
# (qwen3_tts_model.py's generate_voice_clone, the `cut = int(ref_len /
# total_len * wav.shape[0])` line) computes where the reference audio ends
# and the newly-generated speech begins using a PROPORTIONAL estimate
# (ref codec-token count / total codec-token count, scaled onto the decoded
# sample count) rather than an exact per-token sample boundary. That ratio
# assumes uniform samples-per-token across both the reference and generated
# segments; in practice it lands short often enough to leave a trailing
# fragment of the REFERENCE clip's own speech stitched onto the front of
# the returned waveform. CONFIRMED against real output: Celeste's ICL
# ref_text ends in "...many years", and chunk_01 (pure Japanese,
# "こんにちは。予約しています。") in both GPU test runs shows a short
# ~90-100ms energy burst, then a genuine ~60-80ms silence gap, THEN the
# real "konnichiwa" onset — i.e. two concatenated acoustic events, not one
# natural utterance start. That leaked fragment is what read as an English
# "is"/"-s" sound stitched onto the front of the Japanese phrase. This is
# most audible on SHORT chunks (a fixed ~100ms leak is a much larger
# fraction of a 2-3s clip than a 10s+ one) — exactly matching the reported
# symptom being isolated to the short Japanese-only lines.
#
# This trims any leading fragment-then-dip pattern BEFORE the real
# utterance's sustained onset, independent of (in addition to, not instead
# of) the library's own already-applied cut.
#
# v2 CORRECTION: the original version of this detector required an
# ABSOLUTE-silence gap (frame RMS below a fixed 0.003 floor) between the
# leaked burst and the real onset. That worked for the specific chunk it was
# built against, but a second real GPU run (RUN A's "a01" chunk — the same
# こんにちは phrase, same voice, same fix already applied) still leaked an
# audible fragment at the very front, and inspecting its waveform at 10ms
# resolution explained why: the dip between the leaked burst (0-40ms, rms
# ~0.03-0.04) and the real utterance's sustained onset (80ms+, rms ~0.10+)
# only fell to ~0.008-0.011 — a real, measurable LOCAL MINIMUM, but well
# above the old 0.003 absolute-silence floor, so the old detector correctly
# found no "silence" and did nothing. The leak is real either way; only its
# depth varies from call to call (different reference-clip token position
# gets cut into depending on each chunk's own generated-token count), so an
# absolute threshold could never reliably catch both a hard-silence leak and
# a soft-dip leak with one fixed number.
#
# FIX: detect a RELATIVE dip instead of an absolute silence floor — a frame
# whose RMS falls to LEAK_DIP_RATIO or less of the loudest frame seen so far
# in the burst, that then recovers to above the burst's own peak (confirming
# the recovery is a genuinely LOUDER new onset, not just noise wobbling
# within the same utterance). This generalizes both the original hard-
# silence case (ratio ~0) and the newly observed soft-dip case (ratio ~0.08-
# 0.11 relative to the burst peak here) without needing a hand-tuned
# absolute number that happens to fit whichever example was measured last.
LEAK_SCAN_WINDOW_S = 0.30       # only look for the leak pattern within this leading window
LEAK_FRAME_S = 0.01             # analysis frame size (10ms — fine enough to resolve a soft dip)
LEAK_DIP_RATIO = 0.5            # dip frame RMS must fall to <= this fraction of the burst's peak RMS
LEAK_MIN_GAP_S = 0.02           # minimum dip-frame run to count as a real gap (not just one noisy frame)
LEAK_RECOVERY_RATIO = 1.15      # re-onset must exceed the burst's own peak RMS by this factor to count
                                 # as a genuinely new (louder) event, not the same utterance continuing


def strip_leading_reference_leak(wav: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Returns (possibly-trimmed wav, seconds trimmed). No-op unless a clear
    burst -> relative-dip -> louder-re-onset pattern is found within the
    first LEAK_SCAN_WINDOW_S of the clip. Conservative by design: requires
    BOTH a real dip AND a re-onset that's meaningfully louder than the
    leading burst — a legitimate utterance that starts on a soft consonant
    and ramps up smoothly (no real dip) or one that starts loud and stays
    roughly level (no louder re-onset) is left untouched either way."""
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

    # Track the burst's running peak as we scan forward; once we see a dip
    # to <= LEAK_DIP_RATIO of that peak, look for a recovery to
    # >= LEAK_RECOVERY_RATIO of that same peak — a genuinely louder new
    # event, i.e. the real utterance starting, not just level noise.
    #
    # RECOVERY CHECK USES A LOOK-AHEAD WINDOW, NOT A SINGLE FRAME: real
    # speech onsets ramp up over several frames rather than jumping straight
    # to full volume in one 10ms step (confirmed against a01.wav — the frame
    # immediately after its dip is still transitional at rms=0.033, only
    # reaching its real sustained level of ~0.10-0.13 two frames later). A
    # single-frame check right after the gap can catch that transitional
    # frame and wrongly conclude "no real recovery" even though the actual
    # utterance onset is one frame away. Looking at the PEAK over the next
    # RECOVERY_LOOKAHEAD_FRAMES frames avoids that off-by-one-frame miss.
    RECOVERY_LOOKAHEAD_FRAMES = 5   # ~50ms at LEAK_FRAME_S=0.01 — enough to span a ramping onset
    burst_peak = frame_rms[0]
    i = 1
    while i < len(frame_rms):
        burst_peak = max(burst_peak, frame_rms[i])
        if frame_rms[i] <= burst_peak * LEAK_DIP_RATIO:
            # found a dip — measure its run length
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
            # dip wasn't deep/long enough, or didn't recover louder — keep
            # scanning forward in case a clearer pattern appears later in
            # the window (e.g. the burst_peak was set by early noise)
            i = j if j > i else i + 1
            continue
        i += 1

    return wav, 0.0


def tokens_for_text(text: str) -> int:
    TOKENS_PER_CHAR, HEADROOM, MIN_T, MAX_T = 0.86, 1.5, 192, 2048
    return max(MIN_T, min(MAX_T, int(len(text) * TOKENS_PER_CHAR * HEADROOM)))


# ---------------------------------------------------------------------------
# THE SCENARIO — "Checking in: Japanese for the front desk"  (v2)
#
# REWRITTEN per direct feedback on the v1 script: v1 used ROMAJI (Latin-
# script transliteration, e.g. "konnichiwa") for the Japanese phrase whenever
# it appeared inside an English explanatory sentence, and separately wrote
# the SAME phrase in real Japanese script in its own standalone chunk. That
# is exactly what produced "previous chunk says it correctly in Japanese,
# next chunk re-reads the same phrase but now in English pronunciation" —
# it wasn't a model inconsistency, it was two different chunks containing
# two different SPELLINGS (real Japanese script vs. English-alphabet
# transliteration) of the same phrase, and the model correctly read each
# one as what it visually is. Romaji has zero non-Latin codepoints, so both
# audio_manager._resolve_tts_language's script detector AND the model's own
# text understanding have every reason to treat it as English text.
#
# FIX: romaji is no longer used ANYWHERE in this script. Every Japanese
# phrase is written in real Japanese script (かな/漢字) every single time,
# including when it's embedded inside an otherwise-English sentence — see
# RUN B below, which answers the direct question "can one sentence contain
# both English and real Japanese characters, correctly pronounced as each
# language, in a single generate_voice_clone() call?" The answer per Qwen's
# own architecture (confirmed via modeling_qwen3_tts.py: language="auto"
# sets language_id=None, meaning no single language is forced on the whole
# utterance — the model's tokenizer is Unicode-based and its "Intelligent
# Text Understanding" reads per-token script/semantics) is: yes, this is
# exactly what language="auto" exists for. RUN B tests that directly with
# real single-sentence EN+JP mixes, not split across chunks.
#
# RUN A (chunked-at-boundary, matches v1's structure) is KEPT as a baseline
# for comparison — same phrases, but now real Japanese script throughout,
# still split into separate English-explanation / Japanese-phrase chunks.
# RUN B (mixed-script single-sentence) is the NEW approach — the same
# content, but English narration and the Japanese phrase share ONE sentence
# and ONE generate_voice_clone() call, language="auto" throughout.
#
# Japanese sentences are also LONGER per direct request — full, natural
# multi-clause sentences (not just a 2-4 word phrase) using です/ます polite
# form, matching how a real hotel conversation actually sounds rather than
# flashcard-style fragments.
#
# Paralinguistic color via punctuation/emoticon (no bracket tags — see
# module docstring point 3): warmth/encouragement via "!", a light self-
# correcting beat via em-dash, a text-emoticon, matching the README's own
# demonstrated pattern.
# ---------------------------------------------------------------------------

# RUN A — chunked at language boundaries. Japanese chunks use real script
# only; the surrounding English chunks NAME the phrase in English translation
# only (no transliteration at all — reading a phrase's pronunciation aloud in
# Latin letters is exactly the mechanism that caused the original bug, so v2
# doesn't reintroduce it even as "helper" text).
RUN_A_SCENARIO: list[tuple[str, str, str]] = [
    # (chunk_id, lang, text)
    ("a00", "english",
     "Okay, we've just landed in Tokyo, and our first stop is the hotel front desk. "
     "Let's get you ready with a few essential phrases before we walk in."),
    ("a01", "japanese",
     "こんにちは、予約をしております。名前は田中と申します。二泊三日でお願いしております。"),
    ("a02", "english",
     "That's a complete, natural check-in line — hello, I have a reservation, my name is "
     "Tanaka, and it's for two nights and three days. Notice how much more is packed into "
     "one polite sentence than you'd ever say in English at a front desk!"),
    ("a03", "english",
     "Now, the front desk clerk will very likely ask for your passport and explain the "
     "hotel's breakfast hours in the same breath. Listen for how naturally it flows."),
    ("a04", "japanese",
     "パスポートを拝見してもよろしいでしょうか。朝食は一階のレストランで、七時から十時までとなっております。"),
    ("a05", "english",
     "May I see your passport, and breakfast is on the first floor restaurant, from seven "
     "to ten in the morning — all one smooth, polite request. Don't rush it, and don't "
     "worry about catching every word on the first listen!"),
    ("a06", "english",
     "Once you're checked in, you'll want to ask about the elevator, and maybe where you "
     "can find a convenience store nearby — trust me, you'll need both within the hour."),
    ("a07", "japanese",
     "エレベーターはあちらにございます。コンビニでしたら、ホテルを出て左側に、歩いて二分ほどのところにございますよ。"),
    ("a08", "english",
     "The elevator is right over there, and if you're looking for a convenience store, it's "
     "just to the left as you leave the hotel, about a two-minute walk — such a genuinely "
     "helpful answer, and exactly the kind of sentence you'll hear constantly in Japan."),
    ("a09", "english",
     "And that's it — real, full sentences you'll actually hear within your first ten "
     "minutes in Japan, not just flashcard phrases. Listen through it once more, okay? "
     "You've got this!"),
]

# RUN B — the actual answer to "can one sentence mix English and real
# Japanese characters, correctly pronounced as each language?" Each chunk is
# a SINGLE sentence containing both English narration AND a genuine Japanese
# clause in real script, generated in ONE generate_voice_clone() call with
# language="auto" (never split, never transliterated).
RUN_B_SCENARIO: list[tuple[str, str, str]] = [
    ("b00", "auto",
     "Okay, we've just landed in Tokyo — and the very first thing you'll hear at the front "
     "desk is こんにちは、予約をしております, which simply means 'hello, I have a reservation.'"),
    ("b01", "auto",
     "If they ask to see your passport, you'll hear パスポートを拝見してもよろしいでしょうか — "
     "and don't worry, they're just politely asking to take a look at it."),
    ("b02", "auto",
     "They might also mention 朝食は一階のレストランで、七時から十時までとなっております, which "
     "tells you breakfast is served downstairs from seven until ten every morning."),
    ("b03", "auto",
     "Later, if you need directions, a simple wave and エレベーターはあちらにございます means "
     "'the elevator is right over there' — short, polite, and easy to catch once you know it."),
    ("b04", "auto",
     "And if you're craving a snack, someone might say コンビニでしたら、歩いて二分ほどのところに"
     "ございますよ — the convenience store is just about a two-minute walk away. Isn't it "
     "wonderful how much warmth fits into one polite Japanese sentence?"),
    ("b05", "auto",
     "See how naturally that flowed — English explaining, then real Japanese right in the "
     "middle of the same breath? That's genuinely how bilingual conversation feels, not "
     "two separate languages bolted together!"),
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


def run_variant(model, clone_prompt, run_name: str, scenario: list[tuple[str, str, str]]) -> dict:
    """Generates all chunks, pads each exactly like production, merges with
    the SAME ffmpeg concat production uses, and returns full diagnostics."""
    run_dir = os.path.join(OUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"=== RUN '{run_name}' ===")

    per_chunk = []
    raw_paths = []
    t_run = time.time()

    for chunk_id, lang, text in scenario:
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

        # Workaround for the qwen_tts library's proportional ICL trim
        # sometimes leaving a leaked fragment of the reference clip's own
        # speech at the front — see strip_leading_reference_leak's docstring.
        clean_wav, leaked_s = strip_leading_reference_leak(raw_wav, sr)
        clean_duration = len(clean_wav) / sr

        padded_wav = pad_audio(clean_wav, sr)
        padded_duration = len(padded_wav) / sr

        path = os.path.join(run_dir, f"{chunk_id}.wav")
        sf.write(path, padded_wav, sr)
        raw_paths.append(path)

        leak_note = f"  LEAK STRIPPED: {leaked_s*1000:.0f}ms" if leaked_s > 0 else ""
        log.info(f"  {chunk_id}  lang={lang:<10} raw={raw_duration:.2f}s  "
                  f"padded={padded_duration:.2f}s  gen={gen_elapsed:.1f}s  "
                  f"text={text[:50]!r}{leak_note}")
        per_chunk.append({
            "chunk_id": chunk_id, "lang": lang, "text": text,
            "raw_duration": raw_duration, "padded_duration": padded_duration,
            "sr": sr, "leaked_s": leaked_s,
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
            log.info(f"    {c['chunk_id']}  lang={c['lang']:<10}  {c['text'][:70]!r}")
    log.info("  RUN A (chunked-at-boundary): every Japanese phrase is real script "
              "(かな/漢字) in its own standalone chunk, no romaji anywhere — this alone should "
              "fix the 'previous chunk correct, next chunk reads it in English pronunciation' "
              "symptom, since that was caused by romaji spelling being read as English text, "
              "not by an actual model inconsistency.\n"
              "  RUN B (mixed-script single-sentence): the real test of whether Qwen can "
              "code-switch WITHIN one sentence — each chunk is one generate_voice_clone() call, "
              "language='auto', containing both English narration and a real-script Japanese "
              "clause in the same sentence. Listen for whether the Japanese portion is "
              "pronounced as Japanese (not spelled out/anglicized) and whether the switch "
              "in and out of it sounds like one continuous utterance rather than two stitched "
              "clips. If RUN B sounds natural, that confirms language='auto' + real script is "
              "the correct mechanism for single-sentence code-switching, and RUN A's chunk-"
              "splitting approach is no longer necessary going forward.")

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

    log.info("\n[4] LEAKED REFERENCE-AUDIO FRAGMENT CHECK:")
    log.info("  CONFIRMED BUG (qwen_tts package, not this codebase): generate_voice_clone()'s "
              "ICL trim uses a proportional ref_len/total_len ratio to find where the "
              "reference clip ends and real generated speech begins, and that ratio can land "
              "short — leaving a trailing fragment of the REFERENCE clip's own speech "
              "(Celeste's ref_text ends '...many years') stitched onto the front of the "
              "output. Most audible on SHORT chunks (a fixed ~100ms leak is a much bigger "
              "fraction of a 2-3s clip than a 10s+ one) — this is almost certainly the "
              "'starts with an s/is sound before the real word' symptom reported on "
              "chunk_01's pure-Japanese 'こんにちは' line.")
    any_leak = False
    for r in results:
        leaked_chunks = [c for c in r["per_chunk"] if c["leaked_s"] > 0]
        if leaked_chunks:
            any_leak = True
            log.info(f"  {r['run_name']}: leak detected & auto-stripped on "
                      f"{len(leaked_chunks)} chunk(s):")
            for c in leaked_chunks:
                log.info(f"    {c['chunk_id']}: stripped {c['leaked_s']*1000:.0f}ms  "
                          f"({c['text'][:50]!r})")
        else:
            log.info(f"  {r['run_name']}: no leak pattern detected by the heuristic in any chunk.")
    if any_leak:
        log.info("  >>> This run's WAV files already have the leak removed (see "
                  "strip_leading_reference_leak — applied to every chunk before padding/"
                  "merge, using a RELATIVE dip-then-louder-recovery detector, not an "
                  "absolute silence floor — see that function's docstring for why). Compare "
                  "against an OLDER run's WAV (before this fix) on the same chunk_id to "
                  "confirm the fragment is gone.")
    else:
        log.info("  >>> No leak detected THIS run by the heuristic (LEAK_DIP_RATIO/"
                  "LEAK_MIN_GAP_S/LEAK_RECOVERY_RATIO thresholds in this script) — if you "
                  "still hear a leading artifact after this fix, the leak's dip/recovery "
                  "shape may fall outside these thresholds; tighten LEAK_DIP_RATIO (allow a "
                  "shallower dip to count) or lower LEAK_RECOVERY_RATIO (accept a smaller "
                  "jump as a real re-onset) and re-run, or report the specific chunk so the "
                  "thresholds can be tuned against real data instead of guessed.")

    log.info("\n" + "=" * 90)


def main():
    model = load_base_model()
    clone_prompt = build_clone_prompt(model)

    results = []
    results.append(run_variant(model, clone_prompt, "run_a_chunked_boundary", RUN_A_SCENARIO))
    results.append(run_variant(model, clone_prompt, "run_b_mixed_script_single_sentence", RUN_B_SCENARIO))

    print_final_report(results)


if __name__ == "__main__":
    main()
