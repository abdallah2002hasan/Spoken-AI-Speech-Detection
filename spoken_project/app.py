from pathlib import Path
import tempfile
import streamlit as st
from predict import predict

st.set_page_config(page_title="Speech Deepfake Detector", page_icon="🎙️")
st.title("Real vs AI-Generated Speech")
st.write("Upload a WAV/FLAC recording. The detector returns an AI-speech probability.")
model = st.text_input("Model", "outputs/best_model.joblib")
audio = st.file_uploader("Audio file", type=["wav", "flac"])
if audio:
    st.audio(audio)
    suffix = Path(audio.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio.getbuffer()); path = f.name
    if st.button("Analyze speech", type="primary"):
        try:
            result = predict(model, path)
            st.metric("AI-generated probability", f"{100*result['ai_probability']:.1f}%")
            st.success(f"Decision: {result['label']}")
        except Exception as exc:
            st.error(str(exc))

