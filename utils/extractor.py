import pdfplumber
import pytesseract
from PIL import Image

def extract_text_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def extract_text_image(file):
    return pytesseract.image_to_string(Image.open(file))

    