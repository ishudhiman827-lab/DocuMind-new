import pdfplumber
import pytesseract
from PIL import Image

def extract_text_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def extract_text_image(file):
    return pytesseract.image_to_string(Image.open(file))