import pandas as pd
from pathlib import Path
import partitura as pt
import zipfile
import xml.etree.ElementTree as ET


LABEL_COLORS = {
    "scale": "#1f77b4",      # blue
    "arpeggio": "#2ca02c",   # green
    "chord": "#9467bd",      # purple
    "jump": "#d62728",       # red
    "none": "#808080",
}


# Paths
pred_csv = "predictions/cannon-in-d_windowed_gine_predictions.csv"
musicxml_path = "dataset/raw/cannon-in-d.mxl"
output_path = "predictions/cannon-in-d_colored_debug2.musicxml"


def choose_color(pred):
   
    predicted_labels = str(pred["predicted_labels"]).strip()

    if predicted_labels == "none":
        return None

    label = str(pred["top_label"]).strip()
    return LABEL_COLORS.get(label, "#808080")


def load_partitura_note_array(score_path):
    score = pt.load_score(str(score_path))

    try:
        note_array = score.note_array(include_staff=True)
    except TypeError:
        note_array = score.note_array()

    return note_array


def get_partitura_id_map(note_array):
    
    id_map = {}

    for i, note in enumerate(note_array):
        note_id = note["id"]

        if hasattr(note_id, "item"):
            note_id = note_id.item()

        id_map[i] = str(note_id)

    return id_map


def extract_musicxml_from_mxl(mxl_path):

    with zipfile.ZipFile(mxl_path, "r") as zf:
        # Standard .mxl container path
        if "META-INF/container.xml" in zf.namelist():
            container_xml = zf.read("META-INF/container.xml")
            container_root = ET.fromstring(container_xml)

            rootfile = container_root.find(".//{*}rootfile")

            if rootfile is not None:
                full_path = rootfile.attrib.get("full-path")

                if full_path:
                    return zf.read(full_path), full_path

        # Fallback: find the first actual XML score file
        for name in zf.namelist():
            lower = name.lower()

            if lower.endswith(".musicxml") or lower.endswith(".xml"):
                if "container.xml" not in lower:
                    return zf.read(name), name

    raise FileNotFoundError("Could not locate MusicXML inside the .mxl file")


def get_namespace(root):

    if root.tag.startswith("{"):
        uri = root.tag.split("}")[0][1:]
        return {"m": uri}

    return {}


def find_child(elem, tag, ns):
    if ns:
        return elem.find(f"m:{tag}", ns)
    return elem.find(tag)


def find_all(root, tag, ns):
    if ns:
        return root.findall(f".//m:{tag}", ns)
    return root.findall(f".//{tag}")


def set_note_color(note_elem, color, ns):

    note_elem.set("color", color)

    for child_tag in ["notehead", "stem", "accidental", "beam"]:
        if ns:
            children = note_elem.findall(f"m:{child_tag}", ns)
        else:
            children = note_elem.findall(child_tag)

        for child in children:
            child.set("color", color)


# Load predictions
df = pd.read_csv(pred_csv)


# Load Partitura note array and map note_id -> Partitura ID

note_array = load_partitura_note_array(musicxml_path)
id_map = get_partitura_id_map(note_array)

print(f"Prediction rows: {len(df)}")
print(f"Partitura notes: {len(note_array)}")

df["source_note_id"] = df["note_id"].map(id_map)

missing_ids = df["source_note_id"].isna().sum()
print(f"Missing mapped source_note_id rows: {missing_ids}")

# Prediction lookup by Partitura-style ID
pred_lookup = {}

for _, row in df.iterrows():
    sid = row["source_note_id"]

    if pd.isna(sid):
        continue

    pred_lookup[str(sid)] = row


# Load raw MusicXML from .mxl

xml_bytes, inner_xml_path = extract_musicxml_from_mxl(musicxml_path)
root = ET.fromstring(xml_bytes)
ns = get_namespace(root)

print(f"Loaded inner MusicXML file: {inner_xml_path}")


# Color notes using generated Partitura-style XML IDs

matched_xml_notes = 0
colored_notes = 0
left_none = 0
xml_pitched_notes = 0
xml_rests = 0
xml_unpitched_notes = 0
unmatched_xml_notes = 0


part_index = 0
pitched_note_counter = 0

for note_elem in find_all(root, "note", ns):
    # Skip rests
    rest_elem = find_child(note_elem, "rest", ns)
    if rest_elem is not None:
        xml_rests += 1
        continue

    # Skip unpitched notes
    pitch_elem = find_child(note_elem, "pitch", ns)
    if pitch_elem is None:
        xml_unpitched_notes += 1
        continue

    # Generate ID matching Partitura's style
    # Example: p0n0, p0n1, p0n2, ...
    xml_source_id = f"p{part_index}n{pitched_note_counter}"

    pitched_note_counter += 1
    xml_pitched_notes += 1

    if xml_source_id not in pred_lookup:
        unmatched_xml_notes += 1
        continue

    matched_xml_notes += 1

    pred = pred_lookup[xml_source_id]
    color = choose_color(pred)

    if color is None:

        set_note_color(note_elem, "#000000", ns)
        left_none += 1
        continue

    set_note_color(note_elem, color, ns)
    colored_notes += 1

# Save modified MusicXML back to xml for visual in musescore studio 4

output_path = Path(output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)

tree = ET.ElementTree(root)
tree.write(output_path, encoding="utf-8", xml_declaration=True)

print(f"Saved colored score to: {output_path}")
print("---- Match report ----")
print(f"XML pitched notes counted: {xml_pitched_notes}")
print(f"XML rests skipped: {xml_rests}")
print(f"XML unpitched notes skipped: {xml_unpitched_notes}")
print(f"Matched by generated Partitura-style id: {matched_xml_notes}")
print(f"Colored technique notes: {colored_notes}")
print(f"Left as none/black: {left_none}")
print(f"Unmatched XML pitched notes: {unmatched_xml_notes}")
print(f"Unmatched prediction rows: {len(df) - matched_xml_notes}")