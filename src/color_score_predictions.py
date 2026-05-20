import pandas as pd
from pathlib import Path
from music21 import converter, note, chord

LABEL_COLORS = {
    "scale": "#1f77b4",      # blue
    "arpeggio": "#2ca02c",   # green
    "chord": "#9467bd",      # purple
    "jump": "#d62728",       # red
    "none": "#808080",       # gray, currently not used because none stays black
}

# Paths
pred_csv = "predictions/liebestraum-s-541-no-3-in-a-major-liszt_windowed_gine_predictions.csv"
musicxml_path = "dataset/raw/liebestraum-s-541-no-3-in-a-major-liszt.mxl"
output_path = "predictions/liebestraum-s-541-no-3-in-a-major-liszt_colored_debug.musicxml"

# Load predictions
df = pd.read_csv(pred_csv)

# Build lookup from prediction CSV
# Current simple key: measure + pitch + onset
lookup = {}

for _, row in df.iterrows():
    key = (
        int(row["measure"]),
        int(row["pitch"]),
        round(float(row["onset_beat"]), 4),
    )
    lookup.setdefault(key, []).append(row)

# Parse original score
score = converter.parse(musicxml_path)

matched = 0
colored = 0
left_none = 0
unmatched = 0

def choose_color(pred):
    """
    Return color for a prediction row.
    If predicted_labels is none, return None so the note stays black.
    """
    predicted_labels = str(pred["predicted_labels"]).strip()

    if predicted_labels == "none":
        return None

    label = str(pred["top_label"]).strip()
    return LABEL_COLORS.get(label, "#808080")


# Walk through notes in score
for n in score.recurse().notes:

    # Single note
    if isinstance(n, note.Note):
        measure = n.measureNumber - 1  # CSV measure starts at 0
        pitch = n.pitch.midi
        onset = round(float(n.getOffsetInHierarchy(score)), 4)

        key = (measure, pitch, onset)

        if key in lookup:
            matched += 1
            pred = lookup[key].pop(0)

            color = choose_color(pred)

            if color is None:
                left_none += 1
                continue

            n.style.color = color
            colored += 1

        else:
            unmatched += 1

    # Chord object
    elif isinstance(n, chord.Chord):
        measure = n.measureNumber - 1
        onset = round(float(n.getOffsetInHierarchy(score)), 4)

        chord_colors = []

        for p in n.pitches:
            key = (measure, p.midi, onset)

            if key in lookup:
                matched += 1
                pred = lookup[key].pop(0)

                color = choose_color(pred)

                if color is None:
                    left_none += 1
                    continue

                chord_colors.append(color)
                colored += 1
            else:
                unmatched += 1

        # Simple version:
        # music21 colors the whole chord object, not each notehead separately.
        # So use the first detected color in the chord.
        if chord_colors:
            n.style.color = chord_colors[0]

# Make sure output folder exists
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

# Save colored score
score.write("musicxml", fp=output_path)

print(f"Saved colored score to: {output_path}")
print("---- Match report ----")
print(f"Matched notes: {matched}")
print(f"Colored technique notes: {colored}")
print(f"Left as none/black: {left_none}")
print(f"Unmatched notes: {unmatched}")