# Maps each model’s native labels to our common set:
# ["anger","disgust","fear","joy","sadness","surprise","neutral"]

COMMON = ["anger","disgust","fear","joy","sadness","surprise","neutral"]

def zero_scores():
    return {k: 0.0 for k in COMMON}

# --- VIDEO (fer) native labels typically: ['angry','disgust','fear','happy','sad','surprise','neutral']
FER_TO_COMMON = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "joy",
    "sad": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}

# --- AUDIO (superb/wav2vec2-base-superb-er) labels often include: angry, happy, sad, neutral
AUDIO_TO_COMMON = {
    # canonical
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "joy",
    "sad": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
    "calm": "neutral",

    # variants / synonyms that appear in some SER checkpoints
    "fearful": "fear",
    "surprised": "surprise",
    "happiness": "joy",
    "anger": "anger",
    "sadness": "sadness",
}

# --- TEXT (j-hartmann/emotion-english-distilroberta-base) typical labels: anger, disgust, fear, joy, neutral, sadness, surprise
TEXT_TO_COMMON = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
    "calm": "neutral", 
}

def normalize_and_fill(score_dict):
    """Ensure all COMMON keys present; renormalize to sum 1 if total>0."""
    out = zero_scores()
    total = 0.0
    for k, v in score_dict.items():
        if k in out:
            out[k] += float(v)
            total += float(v)
    if total > 0:
        for k in out:
            out[k] /= total
    else:
        out["neutral"] = 1.0
    return out
