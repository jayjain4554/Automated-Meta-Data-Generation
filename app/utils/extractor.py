from pdfminer.high_level import extract_text as extract_pdf
import docx2txt
import pytesseract
from PIL import Image

def extract_text_from_pdf(file_path):
    return extract_pdf(file_path)

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

