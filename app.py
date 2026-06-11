import streamlit as st
from pathlib import Path
from src.parser import load_score, extract_notes, parse_score
from src.graph_builder import build_graph
from src.predict_musicxml_windowed import predict_windowed
from src.color_score_predictions import color_score_from_rows

st.set_page_config(page_title="Piano technique detector", page_icon="🎹")

st.title("GINE: Piano technique detector")
st.write("Upload a MusicXML or MXL score.")

uploaded = st.file_uploader(
    "Upload score",
    type=["musicxml", "xml", "mxl"]
)

if uploaded is not None:
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / uploaded.name
    file_path.write_bytes(uploaded.read())

    st.success(f"Uploaded: {uploaded.name}")

    if st.button("Analyze"):
        result = parse_score(file_path)
        graph = build_graph(result["notes"])

        prediction_rows = predict_windowed(
            score_path=file_path,
            threshold=0.5,
            context=1,
        )

        output_path = Path("outputs") / f"{file_path.stem}_annotated.musicxml"

        colored_path, report = color_score_from_rows(
            prediction_rows=prediction_rows,
            musicxml_path=file_path,
            output_path=output_path,
        )

        st.write("Number of notes:", result["num_notes"])
        st.write("Nodes:", graph.num_nodes)
        st.write("Edges:", graph.edge_index.shape[1])

        st.subheader("Predictions")
        st.dataframe(prediction_rows[:50])

        st.subheader("Coloring Report")
        st.write(report)

        with open(colored_path, "rb") as f:
            st.download_button(
                "Download Annotated Score",
                f,
                file_name=f"{file_path.stem}_annotated.musicxml",
            )