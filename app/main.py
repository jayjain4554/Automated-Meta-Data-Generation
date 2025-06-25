from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
from app.utils.extractor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from app.utils.metadata import generate_metadata

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/generate-metadata")
async def generate(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file.filename.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    elif file.filename.endswith(".txt"):
        text = extract_text_from_txt(file_path)
    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

    # Pass filename and file_path explicitly
    metadata = generate_metadata(text, filename=file.filename, file_path=file_path)
    return metadata

