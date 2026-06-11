import pandas as pd
from pathlib import Path
from collections import defaultdict, deque
import zipfile
import xml.etree.ElementTree as ET


LABEL_COLORS = {
    "scale": "#1f77b4",      # blue
    "arpeggio": "#2ca02c",   # green
    "chord": "#9467bd",      # purple
    "jump": "#d62728",       # red
    "none": "#808080",
}


# Old script-mode paths. These are only used if you run this file directly.
pred_csv = "predictions/cannon-in-d_windowed_gine_predictions.csv"
musicxml_path = "dataset/raw/cannon-in-d.mxl"
output_path = "predictions/cannon-in-d_colored_debug2.musicxml"


def choose_color(pred):
    """
    Color only notes that passed the prediction threshold.

    If predicted_labels == none, the note stays black.
    """
    predicted_labels = str(pred["predicted_labels"]).strip()

    if predicted_labels == "none":
        return None

    label = str(pred["top_label"]).strip()
    return LABEL_COLORS.get(label, "#808080")


def extract_musicxml_bytes(score_path):
    """
    Supports both compressed .mxl files and plain .musicxml/.xml files.
    Returns:
        xml_bytes, inner_name
    """
    score_path = Path(score_path)

    if score_path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(score_path, "r") as zf:
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

    # Plain MusicXML / XML file
    return score_path.read_bytes(), score_path.name


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


def get_text(elem, tag, ns, default=None):
    child = find_child(elem, tag, ns)

    if child is None:
        return default

    if child.text is None:
        return default

    return child.text


def pitch_to_midi(pitch_elem, ns):
    step = get_text(pitch_elem, "step", ns, "C")
    alter = int(get_text(pitch_elem, "alter", ns, 0))
    octave = int(get_text(pitch_elem, "octave", ns, 4))

    step_to_pc = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }

    return (octave + 1) * 12 + step_to_pc[step] + alter


def set_note_color(note_elem, color, ns):
    """
    Set color on the MusicXML note and common child elements.
    MuseScore usually respects color on note/notehead/stem.
    """
    note_elem.set("color", color)

    for child_tag in ["notehead", "stem", "accidental", "beam"]:
        if ns:
            children = note_elem.findall(f"m:{child_tag}", ns)
        else:
            children = note_elem.findall(child_tag)

        for child in children:
            child.set("color", color)


def extract_xml_pitched_notes_with_onsets(root, ns):
    
    xml_notes = []

    divisions = 1
    part_elems = find_all(root, "part", ns)

    for part_elem in part_elems:
        measure_start_div = 0

        if ns:
            measures = part_elem.findall("m:measure", ns)
        else:
            measures = part_elem.findall("measure")

        for measure_elem in measures:
            cursor_div = measure_start_div
            max_cursor_div = measure_start_div
            last_note_onset_div = cursor_div

            for elem in list(measure_elem):
                tag = elem.tag.split("}")[-1]

                if tag == "attributes":
                    div_elem = find_child(elem, "divisions", ns)

                    if div_elem is not None and div_elem.text:
                        divisions = int(div_elem.text)

                elif tag == "backup":
                    duration_div = int(get_text(elem, "duration", ns, 0))
                    cursor_div -= duration_div

                elif tag == "forward":
                    duration_div = int(get_text(elem, "duration", ns, 0))
                    cursor_div += duration_div
                    max_cursor_div = max(max_cursor_div, cursor_div)

                elif tag == "note":
                    rest_elem = find_child(elem, "rest", ns)
                    pitch_elem = find_child(elem, "pitch", ns)
                    chord_elem = find_child(elem, "chord", ns)

                    duration_div = int(get_text(elem, "duration", ns, 0))
                    staff = int(get_text(elem, "staff", ns, 0))

                    if chord_elem is not None:
                        onset_div = last_note_onset_div
                    else:
                        onset_div = cursor_div
                        last_note_onset_div = onset_div

                    if rest_elem is None and pitch_elem is not None:
                        pitch = pitch_to_midi(pitch_elem, ns)
                        onset_beat = onset_div / divisions

                        xml_notes.append(
                            {
                                "elem": elem,
                                "onset_beat": onset_beat,
                                "pitch": pitch,
                                "staff": staff,
                            }
                        )

                    if chord_elem is None:
                        cursor_div += duration_div
                        max_cursor_div = max(max_cursor_div, cursor_div)

            measure_start_div = max_cursor_div

    return xml_notes


def color_score_from_rows(prediction_rows, musicxml_path, output_path):
    df = pd.DataFrame(prediction_rows)

    xml_bytes, inner_xml_path = extract_musicxml_bytes(musicxml_path)
    root = ET.fromstring(xml_bytes)
    ns = get_namespace(root)

    xml_notes = extract_xml_pitched_notes_with_onsets(root, ns)

    # Group predictions by pitch + staff.
    # Instead of exact onset matching, match occurrence order inside each pitch/staff group.
    pred_groups = defaultdict(deque)

    for _, row in df.iterrows():
        key = (
            int(row["pitch"]),
            int(row["staff"]),
        )
        pred_groups[key].append(row)

    matched_xml_notes = 0
    colored_notes = 0
    left_none = 0
    unmatched_xml_notes = 0
    used_prediction_rows = 0

    for xml_note in xml_notes:
        key = (
            int(xml_note["pitch"]),
            int(xml_note["staff"]),
        )

        if key not in pred_groups or not pred_groups[key]:
            unmatched_xml_notes += 1
            continue

        # Take the next unused prediction with the same pitch + staff.
        pred = pred_groups[key].popleft()

        matched_xml_notes += 1
        used_prediction_rows += 1

        color = choose_color(pred)

        if color is None:
            set_note_color(xml_note["elem"], "#000000", ns)
            left_none += 1
        else:
            set_note_color(xml_note["elem"], color, ns)
            colored_notes += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    report = {
        "xml_pitched_notes": len(xml_notes),
        "matched_xml_notes": matched_xml_notes,
        "colored_notes": colored_notes,
        "left_none": left_none,
        "unmatched_xml_notes": unmatched_xml_notes,
        "used_prediction_rows": used_prediction_rows,
        "unmatched_prediction_rows": len(df) - used_prediction_rows,
    }

    return output_path, report