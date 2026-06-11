from pathlib import Path
import pandas as pd

from label_XML_reader import read_labeled_musicxml


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
INPUT_MUSICXML = "dataset/raw_musicXML/musette-in-d-major-johann-sebastian-bach-musette-in-d.musicxml"
OUTPUT_CSV = "dataset/labels/musette-in-d-major.csv"


def main():
    input_path = Path(INPUT_MUSICXML)
    output_path = Path(OUTPUT_CSV)

    if not input_path.exists():
        raise FileNotFoundError(f"Input MusicXML not found: {input_path}")

    rows = read_labeled_musicxml(input_path)

    df = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved labels to: {output_path}")
    print(f"Rows: {len(df)}")

    if df.empty:
        print("Warning: No labeled note rows were found.")
        return

    print("\nFirst 20 rows:")
    print(df.head(20).to_string(index=False))

    label_cols = ["scale", "arpeggio", "chord", "jump"]

    print("\nLabel counts:")
    print(df[label_cols].sum())

    print("\nNone / all-zero count:")
    none_count = (df[label_cols].sum(axis=1) == 0).sum()
    print(none_count)

    print("\nTiming check:")
    timing_cols = [
        "measure",
        "measure_xml",
        "pitch_name",
        "onset_beat",
        "duration_beat",
        "onset_div",
        "duration_div",
        "staff",
        "voice",
        "lyric",
    ]
    print(df[timing_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()