import os, json, math
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from moviepy.editor import VideoFileClip
import cv2
from fer import FER
import librosa
from transformers import pipeline
from faster_whisper import WhisperModel
import math
from typing import Dict
from labels_maps import (
    COMMON, FER_TO_COMMON, AUDIO_TO_COMMON, TEXT_TO_COMMON, normalize_and_fill
)

# --------------------------
# Config
# --------------------------
FRAME_STRIDE_SEC = 0.25         # sample one frame every 0.25s
AUDIO_CHUNK_SEC  = 0.25         # classify audio every 5s
W_VIDEO, W_AUDIO, W_TEXT = 0.40, 0.40, 0.20
ASR_MODEL_SIZE = "small"       # "tiny"|"base"|"small"|"medium"|"large-v3"
DEVICE = "cpu"                 # set "cuda" if you have GPU for HF pipelines
OUT_DIR = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# Helpers
# --------------------------


def confidence_of(scores: Dict[str,float]) -> float:
    p = [max(1e-12, scores[k]) for k in COMMON]
    s = sum(p)
    p = [v/s for v in p]
    H = -sum(v*math.log(v) for v in p)
    Hmax = math.log(len(COMMON))
    return max(0.0, 1.0 - H/Hmax)

def softmax_dict(d):
    vals = np.array(list(d.values()), dtype=np.float32)
    exps = np.exp(vals - vals.max())
    probs = exps / exps.sum()
    return {k: float(p) for k, p in zip(d.keys(), probs)}

def add_scores(acc, add):
    for k in acc:
        acc[k] += add.get(k, 0.0)
    return acc

def avg_scores(acc, n):
    if n == 0:
        return acc
    return {k: v / n for k, v in acc.items()}

def topk(scores, k=3):
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

def map_scores(native_scores, mapping):
    out = {c:0.0 for c in COMMON}
    for label, score in native_scores.items():
        key = label.lower()
        if key in mapping:
            out[mapping[key]] += score
    # normalize
    return normalize_and_fill(out)

# ---------- RAVDESS filename parsing ----------
RAVDESS_EMO_CODE_TO_LABEL = {
    "01": "neutral", "02": "calm", "03": "joy", "04": "sadness",
    "05": "anger",  "06": "fear",  "07": "disgust", "08": "surprise"
}

def parse_ravdess_meta_from_name(filename: str):
    """
    RAVDESS: MM-VC-EE-II-SS-RR-AA.ext
    We return {'gt': <COMMON label>, 'modality': MM, 'vocal': VC}
    Calm is mapped to neutral to match COMMON.
    """
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    modality, vocal, emo_code = parts[0], parts[1], parts[2]
    lab = RAVDESS_EMO_CODE_TO_LABEL.get(emo_code)
    if lab is None:
        return None
    if lab == "calm":
        lab = "neutral"
    return {"gt": lab, "modality": modality, "vocal": vocal}




# ---------- simple confusion counter ----------
def _blank_confusion():
    return {c: {c2: 0 for c2 in COMMON} for c in COMMON}

def _update_conf(conf, y_true, y_pred):
    if y_true in COMMON and y_pred in COMMON:
        conf[y_true][y_pred] += 1

# ---------- evaluate a folder of videos ----------
def evaluate_dir(dir_path: str, pattern_suffix=".mp4"):
    files = []
    for root, _, fnames in os.walk(dir_path):
        for f in fnames:
            if f.lower().endswith(pattern_suffix):
                files.append(os.path.join(root, f))
    files.sort()
    if not files:
        print(f"[Eval] No {pattern_suffix} files under: {dir_path}")
        return

    # Confusion matrices
    def _blank_confusion():
        return {c: {c2: 0 for c2 in COMMON} for c in COMMON}
    def _update_conf(conf, y_true, y_pred):
        if y_true in COMMON and y_pred in COMMON:
            conf[y_true][y_pred] += 1

    conf_video = _blank_confusion()
    conf_audio = _blank_confusion()
    conf_text  = _blank_confusion()
    conf_fused = _blank_confusion()

    rows = []
    totals = 0
    correct_v = correct_a = correct_t = correct_f = 0

    for path in files:
        meta = parse_ravdess_meta_from_name(path)
        if not meta:
            print(f"[Eval] Skip (bad name): {os.path.basename(path)}")
            continue

        # If we ONLY want full AV speech, then uncomment:
        # if meta["modality"] != "01" or meta["vocal"] != "01":
        #     continue

        gt = meta["gt"]
        print(f"[Eval] {os.path.basename(path)}  (expected={gt})")

        r = analyze_one(path)

        vpred, apred, tpred, fpred = r["video_pred"], r["audio_pred"], r["text_pred"], r["fused_pred"]

        _update_conf(conf_video, gt, vpred)
        _update_conf(conf_audio, gt, apred)
        _update_conf(conf_text,  gt, tpred)
        _update_conf(conf_fused, gt, fpred)

        totals += 1
        cv = int(vpred == gt); ca = int(apred == gt); ct = int(tpred == gt); cf = int(fpred == gt)
        correct_v += cv; correct_a += ca; correct_t += ct; correct_f += cf

        # Base columns in the exact order you requested
        row = {
            "file": path,
            "expected": gt,
            "fused_pred": fpred,
            "video_pred": vpred,
            "audio_pred": apred,
            "text_pred":  tpred,
            "fused_accuracy": cf,
            "video_accuracy": cv,
            "audio_accuracy": ca,
            "text_accuracy":  ct,
        }

        # Append remaining probability columns (fused_*, video_*, audio_*, text_*)
        row.update({f"fused_{k}":  float(v) for k, v in r["fused_scores"].items()})
        row.update({f"video_{k}":  float(v) for k, v in r["video_scores"].items()})
        row.update({f"audio_{k}":  float(v) for k, v in r["audio_scores"].items()})
        row.update({f"text_{k}":   float(v) for k, v in r["text_scores"].items()})

        rows.append(row)

    if totals == 0:
        print("[Eval] No valid files after filtering.")
        return

    acc_video = correct_v / totals
    acc_audio = correct_a / totals
    acc_text  = correct_t / totals
    acc_fused = correct_f / totals

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Files evaluated: {totals}")
    print(f"Video  accuracy: {acc_video:.3f}")
    print(f"Audio  accuracy: {acc_audio:.3f}")
    print(f"Text   accuracy: {acc_text:.3f}")
    print(f"FUSED  accuracy: {acc_fused:.3f}")

    # --- Save CSV in the requested order ---
    import csv
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "batch_results.csv")

    # Build header: base columns first, then all remaining prob columns (sorted for stability)
    base_cols = [
        "file", "expected",
        "fused_pred", "video_pred", "audio_pred", "text_pred",
        "fused_accuracy", "video_accuracy", "audio_accuracy", "text_accuracy",
    ]
    # collect remaining keys from the first row to lock order deterministically
    prob_cols = [c for c in rows[0].keys() if c not in base_cols]
    # keep fused_* first, then video_*, audio_*, text_* (each alpha inside)
    prob_cols = sorted([c for c in prob_cols if c.startswith("fused_")]) + \
                sorted([c for c in prob_cols if c.startswith("video_")]) + \
                sorted([c for c in prob_cols if c.startswith("audio_")]) + \
                sorted([c for c in prob_cols if c.startswith("text_")])

    header = base_cols + prob_cols

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

    print(f"[Eval] Saved per-file CSV → {csv_path}")

    # --- Save confusions + accuracies JSON ---
    conf_path = os.path.join(OUT_DIR, "confusions.json")
    with open(conf_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video": conf_video,
                "audio": conf_audio,
                "text":  conf_text,
                "fused": conf_fused,
                "accuracy": {
                    "video": acc_video, "audio": acc_audio,
                    "text": acc_text,   "fused": acc_fused
                }
            },
            f, indent=2
        )
    print(f"[Eval] Saved confusion counts → {conf_path}")


# --------------------------
# Video emotion (facial)
# --------------------------
def analyze_video_emotions(video_path):
    clip = VideoFileClip(video_path)
    detector = FER(mtcnn=False)  # MTCNN face detector; 

    duration = clip.duration
    fps = clip.fps or 25
    frame_scores_acc = {c:0.0 for c in COMMON}
    n = 0

    for t in np.arange(0, duration, FRAME_STRIDE_SEC):
        frame = clip.get_frame(t)  # RGB
        # FER expects RGB np.array
        results = detector.detect_emotions(frame)  # list of faces with 'emotions'
        # Aggregate all faces in the frame (usually one person)
        if results:
            per_frame = {}
            for r in results:
                emo = r.get("emotions", {})
                # emo example keys: 'angry','disgust','fear','happy','sad','surprise','neutral'
                for k,v in emo.items():
                    per_frame[k] = per_frame.get(k, 0.0) + float(v)
            # softmax over aggregated face logits for that frame
            frame_native_scores = softmax_dict(per_frame) if per_frame else {}
            frame_common = map_scores(frame_native_scores, FER_TO_COMMON)
            frame_scores_acc = add_scores(frame_scores_acc, frame_common)
            n += 1

    clip.close()
    return avg_scores(frame_scores_acc, n if n>0 else 1)

# --------------------------
# Audio extraction
# --------------------------
def extract_audio_array(video_path, sr=16000):
    """Extract mono audio samples from video using MoviePy, no librosa needed for MP4."""
    from moviepy.editor import VideoFileClip
    import numpy as np

    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        return np.array([], dtype=np.float32), sr

    # Get audio as array at target sampling rate
    # MoviePy returns float32 in [-1, 1], shape (N, 2) for stereo
    try:
        arr = clip.audio.to_soundarray(fps=sr)  # (N, 1 or 2)
        clip.close()
        if arr.ndim == 2:
            arr = arr.mean(axis=1)  # convert to mono
        return arr.astype("float32"), sr
    finally:
        try:
            clip.close()
        except:
            pass


# --------------------------
# Audio emotion (SER)
# --------------------------
def _ser_probe(video_path):
    import numpy as np
    from transformers import pipeline

    y, sr = extract_audio_array(video_path, sr=16000)
    if y.size == 0:
        print("[SER PROBE] No audio.")
        return

    # take a 2s slice from the middle
    mid = len(y) // 2
    win = sr * 2
    start = max(0, mid - win // 2)
    chunk = y[start:start+win].astype(np.float32)

    print(f"[SER PROBE] slice len={len(chunk)}, rms={(float((chunk**2).mean())**0.5):.6e}, max={float(np.abs(chunk).max()):.4f}")

    ser = pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device=-1 if DEVICE == "cpu" else 0,
        return_all_scores=True,
    )
    preds = ser({"array": chunk, "sampling_rate": 16000})
    if isinstance(preds, list) and preds and isinstance(preds[0], list):
        preds = preds[0]
    print("[SER PROBE] raw preds:", preds[:8])

def analyze_audio_emotions(video_path):
    import numpy as np
    from transformers import pipeline

    y, sr = extract_audio_array(video_path, sr=16000)
    if y.size == 0:
        return {c: (1.0 if c == "neutral" else 0.0) for c in COMMON}

    # --- parameters ---
    chunk_sec = 2.0  # shorter chunks help on brief clips
    chunk_len = int(chunk_sec * sr)

    # Dynamic silence threshold: 5% of global RMS (min clamp)
    global_rms = float(np.sqrt((y.astype(np.float32) ** 2).mean() + 1e-12))
    rms_floor = max(1e-6, 0.05 * global_rms)

    ser = pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device=-1 if DEVICE == "cpu" else 0,
        return_all_scores=True,
    )


    total = {c: 0.0 for c in COMMON}
    n = 0

    # Debug: show first 2 chunk stats
    dbg_shown = 0

    for start in range(0, len(y), chunk_len):
        end = min(start + chunk_len, len(y))
        if end - start < int(0.75 * sr):
            continue
        chunk = y[start:end].astype(np.float32)

        rms = float(np.sqrt((chunk ** 2).mean() + 1e-12))
        if dbg_shown < 2:
            print(f"[SER] {start/sr:.2f}-{end/sr:.2f}s rms={rms:.6e} (floor={rms_floor:.6e})")
        if rms < rms_floor:
            total["neutral"] += 1.0
            n += 1
            if dbg_shown < 2:
                print("       -> treated as silence → neutral")
                dbg_shown += 1
            continue

        preds = ser({"array": chunk, "sampling_rate": sr})
        if isinstance(preds, list) and preds and isinstance(preds[0], list):
            preds = preds[0]

        if not preds:
            total["neutral"] += 1.0
            n += 1
            if dbg_shown < 2:
                print("       -> empty preds → neutral")
                dbg_shown += 1
            continue

        # Convert model labels to our space
        native = {p["label"].lower(): float(p["score"]) for p in preds}

        # fold synonyms before mapping
        if "fearful" in native:
            native["fear"] = native.pop("fearful") + native.get("fear", 0.0)
        if "surprised" in native:
            native["surprise"] = native.pop("surprised") + native.get("surprise", 0.0)
        if "happiness" in native:
            native["happy"] = native.pop("happiness") + native.get("happy", 0.0)
        if "calm" in native:
            native["neutral"] = native.pop("calm") + native.get("neutral", 0.0)

        
        mapped = map_scores(native, AUDIO_TO_COMMON)
        total = add_scores(total, mapped)
        n += 1

        if dbg_shown < 2:
            print("       -> preds:", preds[:4])
            dbg_shown += 1

    if n == 0:
        return {c: (1.0 if c == "neutral" else 0.0) for c in COMMON}
    return {k: v / n for k, v in total.items()}


# --------------------------
# Speech-to-text (Whisper) + Text emotion
# --------------------------
def transcribe_text(video_path):
    # Faster-Whisper runs local; set device="cuda" if available
    model = WhisperModel(ASR_MODEL_SIZE, device=DEVICE, compute_type="int8" if DEVICE=="cpu" else "float16")
    segments, info = model.transcribe(video_path, beam_size=5, vad_filter=True)
    text = " ".join([seg.text.strip() for seg in segments if seg.text])
    return text.strip()

def analyze_text_emotions(text):
    from transformers import pipeline
    from labels_maps import TEXT_TO_COMMON, COMMON, normalize_and_fill

    # Empty or super-short transcript → fall back to neutral
    if not text or not text.strip():
        return {c: (1.0 if c == "neutral" else 0.0) for c in COMMON}

    clf = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,                # request all labels
        device=-1 if DEVICE == "cpu" else 0
    )
    preds = clf(text)

    # Normalize output shape:
    # - if it's [[{...},{...}]] take the first inner list
    # - if it's [{...},{...}] use as-is
    if isinstance(preds, list) and preds and isinstance(preds[0], list):
        preds = preds[0]

    # Now preds should be a list of dicts with keys 'label' and 'score'
    native = {}
    for p in preds:
        lbl = str(p.get("label", "")).lower()
        sc = float(p.get("score", 0.0))
        if lbl:
            native[lbl] = native.get(lbl, 0.0) + sc

    native = normalize_and_fill(native)

    out = {c: 0.0 for c in COMMON}
    for lbl, sc in native.items():
        mapped = TEXT_TO_COMMON.get(lbl, "neutral")
        out[mapped] += sc
    return normalize_and_fill(out)

# --------------------------
# Analsyse one video file
# --------------------------
def analyze_one(video_path: str):
    video_scores = analyze_video_emotions(video_path)
    audio_scores = analyze_audio_emotions(video_path)
    transcript   = transcribe_text(video_path)
    text_scores  = analyze_text_emotions(transcript)

    fused_scores, fused_pred = fuse_scores(
        video_scores, audio_scores, text_scores, transcript=transcript
    )

    return {
        "video_scores": video_scores,
        "audio_scores": audio_scores,
        "text_scores":  text_scores,
        "fused_scores": fused_scores,
        "video_pred": max(video_scores, key=video_scores.get),
        "audio_pred": max(audio_scores, key=audio_scores.get),
        "text_pred":  max(text_scores,  key=text_scores.get),
        "fused_pred": fused_pred,
        "transcript": transcript,
    }

# --------------------------
# Fusion & Summary
# --------------------------
def fuse_scores(video_s, audio_s, text_s, transcript=None, wv=W_VIDEO, wa=W_AUDIO, wt=W_TEXT):
    base = {"video": wv, "audio": wa, "text": wt}

    # confidences
    cv, ca, ct = confidence_of(video_s), confidence_of(audio_s), confidence_of(text_s)

    # --- scale TEXT trust by transcript length and neutrality ---
    words = 0 if not transcript else len(transcript.split())
    short_factor = min(1.0, words / 8.0)  # need ~8+ words for full trust
    t_top = max(text_s, key=text_s.get)
    if t_top == "neutral":
        ct *= 0.20 * short_factor     # strongly damp neutral text on short clips
    else:
        ct *= 0.6 * short_factor + 0.4

    # confidence-adjusted base weights
    exp = 1.5
    adj = {
        "video": base["video"] * (cv + 1e-3)**exp,
        "audio": base["audio"] * (ca + 1e-3)**exp,
        "text":  base["text"]  * (ct + 1e-3)**exp,
    }

    # prelim normalized weights
    Z = adj["video"] + adj["audio"] + adj["text"]
    wv2, wa2, wt2 = adj["video"]/Z, adj["audio"]/Z, adj["text"]/Z

    # --- floors/caps: don't let weak text dominate; give face/voice a floor if non-neutral ---
    v_top = max(video_s, key=video_s.get)
    a_top = max(audio_s, key=audio_s.get)

    wt2 = min(wt2, 0.35)  # text cap
    if v_top != "neutral" and cv > 0.05:
        wv2 = max(wv2, 0.20)  # vision floor if any non-neutral signal
    if a_top != "neutral" and ca > 0.10:
        wa2 = max(wa2, 0.30)  # audio floor if any non-neutral signal

    # re-normalize
    Z = wv2 + wa2 + wt2
    wv2, wa2, wt2 = wv2/Z, wa2/Z, wt2/Z

    # fuse
    fused = {c: wv2*video_s[c] + wa2*audio_s[c] + wt2*text_s[c] for c in COMMON}

    # --- agreement boost (two modalities same non-neutral top) ---
    t_top = max(text_s, key=text_s.get)
    tops = [v_top, a_top, t_top]
    for lab in COMMON:
        if lab != "neutral" and tops.count(lab) >= 2:
            fused[lab] += 0.20   # stronger boost
            break

    # --- neutral guard: if at least TWO modalities are confidently NON-neutral, cap neutral ---
    non_neutral_modalities = 0
    if v_top != "neutral" and cv > 0.10: non_neutral_modalities += 1
    if a_top != "neutral" and ca > 0.10: non_neutral_modalities += 1
    if t_top != "neutral" and ct > 0.10: non_neutral_modalities += 1

    if non_neutral_modalities >= 2:
        fused["neutral"] = min(fused["neutral"], 0.15)

    # normalize
    S = sum(max(0.0, v) for v in fused.values())
    if S > 0:
        for k in fused:
            fused[k] = max(0.0, fused[k]) / S
    else:
        fused = {c:(1.0 if c=="neutral" else 0.0) for c in COMMON}

    final = max(fused.items(), key=lambda x: x[1])[0]

    # debug
    print(f"[Fusion] conf(video,audio,text) = {cv:.3f}, {ca:.3f}, {ct:.3f} (ct post-scale)")
    print(f"[Fusion] weights(adjusted) = V:{wv2:.3f} A:{wa2:.3f} T:{wt2:.3f}")
    print("[Fusion] tops:", f"video={v_top}", f"audio={a_top}", f"text={t_top}")
    print("[Fusion] fused top3:", sorted(fused.items(), key=lambda x: x[1], reverse=True)[:3])

    return fused, final


def make_summary(final_label, fused, video_s, audio_s, text_s, transcript):
    # Simple, readable one-liner + modality hints
    top3 = ", ".join([f"{k}:{v:.2f}" for k,v in topk(fused,3)])
    hints = []
    vid_top = topk(video_s,1)[0][0]
    aud_top = topk(audio_s,1)[0][0]
    txt_top = topk(text_s,1)[0][0]
    hints.append(f"face≈{vid_top}")
    hints.append(f"voice≈{aud_top}")
    hints.append(f"text≈{txt_top}")
    base = f"Overall emotion: **{final_label.upper()}** (top mix → {top3}; {', '.join(hints)})."
    # Add transcript snippet if short
    snippet = transcript[:160].strip()
    if snippet:
        base += f" Transcript snippet: “{snippet}”"
    return base

# --------------------------
# Main
# --------------------------
def run(video_path):
    print(f"[Phase1] Analyzing: {video_path}")
    _ser_probe(video_path)
    video_scores = analyze_video_emotions(video_path)
    print(" - Video done")
    audio_scores = analyze_audio_emotions(video_path)
    print(" - Audio done")
    transcript = transcribe_text(video_path)
    print(" - Transcription done")
    text_scores = analyze_text_emotions(transcript)
    print(" - Text emotion done")

    fused, final_label = fuse_scores(video_scores, audio_scores, text_scores, transcript=transcript)
    summary = make_summary(final_label, fused, video_scores, audio_scores, text_scores, transcript)

    result = {
        "video_emotion_scores": video_scores,
        "audio_emotion_scores": audio_scores,
        "text_emotion_scores": text_scores,
        "fused_scores": fused,
        "final_label": final_label,
        "transcript": transcript,
        "summary": summary
    }
    out_path = os.path.join(OUT_DIR, "result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n==== EMOTION SUMMARY ====")
    print(summary)
    print(f"\nSaved JSON → {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="Path to a single input video (with audio)")
    ap.add_argument("--dir",   help="Evaluate all .mp4 under this folder (recursively)")
    args = ap.parse_args()

    if args.dir:
        evaluate_dir(args.dir)
    elif args.video:
        # single-file flow (writes outputs/result.json like before)
        print(f"[Phase1] Analyzing: {args.video}")
        video_scores = analyze_video_emotions(args.video)
        print(" - Video done")
        audio_scores = analyze_audio_emotions(args.video)
        print(" - Audio done")
        transcript   = transcribe_text(args.video)
        print(" - Transcription done")
        text_scores  = analyze_text_emotions(transcript)
        print(" - Text emotion done")

        fused, final_label = fuse_scores(video_scores, audio_scores, text_scores, transcript=transcript)
        summary = make_summary(final_label, fused, video_scores, audio_scores, text_scores, transcript)

        result = {
            "video_emotion_scores": video_scores,
            "audio_emotion_scores": audio_scores,
            "text_emotion_scores":  text_scores,
            "fused_scores": fused,
            "final_label": final_label,
            "transcript": transcript,
            "summary": summary
        }
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, "result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("\n==== EMOTION SUMMARY ====")
        print(summary)
        print(f"\nSaved JSON → {out_path}")
    else:
        ap.error("Provide either --video <path> or --dir <folder>")


