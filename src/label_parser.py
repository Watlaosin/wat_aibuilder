LABELS = ["scale", "arpeggio", "chord", "jump"]

LYRIC_DECODE = {

    "S" : [1, 0, 0, 0], # scale
    "A" : [0, 1, 0, 0], # arpeggio
    "C" : [0, 0, 1, 0], # chord
    "J" : [0, 0, 0, 1],  # jump
    "N" : [0, 0, 0, 0], # none 

    "SJ" : [1, 0, 0, 1], # scale + jump
    "JS" : [1, 0, 0, 1], # jump + scale

    "AJ" : [0, 1, 0, 1], # arpeggio + jump
    "JA" : [0, 1, 0, 1], # jump + arpeggio

    "CJ" : [0, 0, 1, 1], # chord + jump
    "JC" : [0, 0, 1, 1], # jump + chord

}

def parse_label_code(code):
    
    note = code.strip().upper()

    if note not in LYRIC_DECODE:
         raise ValueError(f"Unknown lyric note: {code}")
    
    return LYRIC_DECODE[note]

def assign_labels_to_group(label_text, num_notes):
    label_text = label_text.strip().upper()

    if "/" not in label_text:
        vector = parse_label_code(label_text)
        return [vector.copy() for _ in range(num_notes)]

    parts = [part.strip().upper() for part in label_text.split("/")]

    if len(parts) > num_notes:
        raise ValueError(
            f"Too many labels: got {len(parts)} labels for {num_notes} notes: {label_text}"
        )

    while len(parts) < num_notes:
        parts.append("N")

    return [parse_label_code(part) for part in parts]


