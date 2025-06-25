import streamlit as st
import requests
import json

st.set_page_config(page_title="Metadata Generator", layout="centered")

st.title("📄 Automated Metadata Generator")
st.write("Upload a document to extract metadata using NLP.")

uploaded_file = st.file_uploader("Choose a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")

    if st.button("🔍 Generate Metadata"):
        with st.spinner("Processing..."):
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post("http://localhost:8000/generate-metadata", files=files)

            if response.status_code == 200:
                metadata = response.json()

                st.subheader("📑 Metadata Output")
                st.json(metadata)

                st.subheader("📥 Download Metadata")
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(metadata, indent=2),
                    file_name="metadata.json",
                    mime="application/json"
                )
            else:
                st.error("❌ Failed to generate metadata. Please check backend logs.")
