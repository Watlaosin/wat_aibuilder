import xml.etree.ElementTree as ET
from pathlib import Path

from label_parser import assign_labels_to_group


def get_namespace(root):
    if root.tag.startswith("{"):
        namespace_uri = root.tag.split("}")[0][1:]
        return namespace_uri
    return None


def tag(name, namespace_uri):
    if namespace_uri:
        return f"{{{namespace_uri}}}{name}"
    return name


def child_text(element, child_name, namespace_uri, default=""):
    child = element.find(tag(child_name, namespace_uri))

    if child is None or child.text is None:
        return default

    return child.text.strip()


def get_duration(element, namespace_uri):
    duration = element.find(tag("duration", namespace_uri))

    if duration is None or duration.text is None:
        return 0

    return int(duration.text)


def get_move_duration(element, namespace_uri):
    duration = element.find(tag("duration", namespace_uri))

    if duration is None or duration.text is None:
        return 0

    return int(duration.text)


def get_divisions_from_measure(measure, namespace_uri, current_divisions):
    attributes = measure.find(tag("attributes", namespace_uri))

    if attributes is None:
        return current_divisions

    divisions = attributes.find(tag("divisions", namespace_uri))

    if divisions is None or divisions.text is None:
        return current_divisions

    return int(divisions.text)


def pitch_to_midi(note_element, namespace_uri):
    pitch = note_element.find(tag("pitch", namespace_uri))

    if pitch is None:
        return None

    step = child_text(pitch, "step", namespace_uri)
    octave = int(child_text(pitch, "octave", namespace_uri))

    alter_text = child_text(pitch, "alter", namespace_uri, default="0")
    alter = int(float(alter_text))

    step_to_semitone = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }

    return 12 * (octave + 1) + step_to_semitone[step] + alter


def pitch_to_name(note_element, namespace_uri):
    pitch = note_element.find(tag("pitch", namespace_uri))

    if pitch is None:
        return "REST"

    step = child_text(pitch, "step", namespace_uri)
    octave = child_text(pitch, "octave", namespace_uri)
    alter_text = child_text(pitch, "alter", namespace_uri, default="0")
    alter = int(float(alter_text))

    if alter == 1:
        accidental = "#"
    elif alter == -1:
        accidental = "b"
    else:
        accidental = ""

    return f"{step}{accidental}{octave}"


def get_lyrics(note_element, namespace_uri):
    lyrics = []

    for lyric in note_element.findall(tag("lyric", namespace_uri)):
        text_element = lyric.find(tag("text", namespace_uri))

        if text_element is not None and text_element.text:
            lyrics.append(text_element.text.strip())

    return lyrics


def get_staff(note_element, namespace_uri):
    return child_text(note_element, "staff", namespace_uri, default="")


def get_voice(note_element, namespace_uri):
    return child_text(note_element, "voice", namespace_uri, default="")


def read_labeled_musicxml(musicxml_path):

    musicxml_path = Path(musicxml_path)

    tree = ET.parse(musicxml_path)
    root = tree.getroot()

    namespace_uri = get_namespace(root)

    rows = []
    xml_note_index = 0

    parts = root.findall(f".//{tag('part', namespace_uri)}")

    print(f"Found {len(parts)} part(s).")

    for part_index, part in enumerate(parts):
        part_id = part.attrib.get("id", f"P{part_index}")

        measures = part.findall(tag("measure", namespace_uri))

        current_divisions = 1

        for measure_index, measure in enumerate(measures):
            measure_number_raw = measure.attrib.get("number", str(measure_index))

            measure_number = measure_index

            current_divisions = get_divisions_from_measure(
                measure=measure,
                namespace_uri=namespace_uri,
                current_divisions=current_divisions,
            )

            cursor_div = 0
            last_note_onset_div = 0
            current_group = []

            def flush_group():
                nonlocal current_group, rows

                if not current_group:
                    return

                pitched_notes = [
                    note for note in current_group
                    if note["pitch_midi"] is not None
                ]

                if not pitched_notes:
                    current_group = []
                    return

                lyric_texts = []

                for note in pitched_notes:
                    lyric_texts.extend(note["lyrics"])

                label_text = lyric_texts[0] if lyric_texts else "N"

                # bottom-to-top
                pitched_notes_sorted = sorted(
                    pitched_notes,
                    key=lambda note: note["pitch_midi"],
                )

                try:
                    label_vectors = assign_labels_to_group(
                        label_text=label_text,
                        num_notes=len(pitched_notes_sorted),
                    )

                except ValueError as e:
                    print("\n========== LABEL PARSE ERROR ==========")
                    print(f"Error: {e}")

                    print("\nLabel text:")
                    print(repr(label_text))

                    print("\nNumber of notes in this parser group:")
                    print(len(pitched_notes_sorted))

                    print("\nNotes in this parser group, bottom-to-top:")
                    for i, note in enumerate(pitched_notes_sorted):
                        print(
                            f"  note {i}: "
                            f"measure={note['measure']}, "
                            f"measure_xml={note['measure_xml']}, "
                            f"xml_note_index={note['xml_note_index']}, "
                            f"pitch_name={note['pitch_name']}, "
                            f"pitch_midi={note['pitch_midi']}, "
                            f"staff={note['staff']}, "
                            f"voice={note['voice']}, "
                            f"onset_div={note['onset_div']}, "
                            f"duration_div={note['duration_div']}, "
                            f"onset_beat={note['onset_beat']}, "
                            f"duration_beat={note['duration_beat']}, "
                            f"lyrics={note['lyrics']}"
                        )

                    print("=======================================\n")
                    raise

                for note, label_vector in zip(pitched_notes_sorted, label_vectors):
                    row = {
                        "part_index": note["part_index"],
                        "part_id": note["part_id"],
                        "measure": note["measure"],
                        "measure_xml": note["measure_xml"],
                        "xml_note_index": note["xml_note_index"],
                        "pitch_name": note["pitch_name"],
                        "pitch_midi": note["pitch_midi"],
                        "staff": note["staff"],
                        "voice": note["voice"],
                        "lyric": label_text,

                        # timing fields for graph builder
                        "onset_div": note["onset_div"],
                        "duration_div": note["duration_div"],
                        "onset_beat": note["onset_beat"],
                        "duration_beat": note["duration_beat"],
                        "divs_pq": note["divs_pq"],

                        # labels
                        "scale": label_vector[0],
                        "arpeggio": label_vector[1],
                        "chord": label_vector[2],
                        "jump": label_vector[3],
                    }

                    rows.append(row)

                current_group = []

            for element in measure:
                if element.tag == tag("backup", namespace_uri):
                    flush_group()
                    cursor_div -= get_move_duration(element, namespace_uri)
                    continue

                if element.tag == tag("forward", namespace_uri):
                    flush_group()
                    cursor_div += get_move_duration(element, namespace_uri)
                    continue

                if element.tag != tag("note", namespace_uri):
                    continue

                is_chord_continuation = (
                    element.find(tag("chord", namespace_uri)) is not None
                )

                duration_div = get_duration(element, namespace_uri)

                if is_chord_continuation:
                    onset_div = last_note_onset_div
                else:
                    flush_group()
                    onset_div = cursor_div
                    last_note_onset_div = onset_div

                onset_beat = measure_number * 4.0 + (onset_div / current_divisions)
                duration_beat = duration_div / current_divisions

                note_info = {
                    "part_index": part_index,
                    "part_id": part_id,
                    "measure": measure_number,
                    "measure_xml": measure_number_raw,
                    "xml_note_index": xml_note_index,
                    "pitch_name": pitch_to_name(element, namespace_uri),
                    "pitch_midi": pitch_to_midi(element, namespace_uri),
                    "staff": get_staff(element, namespace_uri),
                    "voice": get_voice(element, namespace_uri),
                    "lyrics": get_lyrics(element, namespace_uri),
                    "onset_div": onset_div,
                    "duration_div": duration_div,
                    "onset_beat": onset_beat,
                    "duration_beat": duration_beat,
                    "divs_pq": current_divisions,
                }

                current_group.append(note_info)
                xml_note_index += 1

                if not is_chord_continuation:
                    cursor_div += duration_div

            flush_group()

    return rows