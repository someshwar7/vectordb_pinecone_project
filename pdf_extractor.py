# modules/pdf_extractor.py
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os
from typing import List, Dict
from .utils import setup_logging, clean_text

logger = setup_logging()

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class PDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.text_data = []
        self.images = []

    def extract_text(self) -> List[Dict]:
        """Extract text from PDF pages using pdfplumber."""
        logger.info(f"Extracting text from {self.pdf_path}")

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    cleaned_text = clean_text(text)
                    if cleaned_text:
                        self.text_data.append({
                            'page_number': page_num,
                            'content': cleaned_text,
                            'type': 'text'
                        })

        logger.info(f"Extracted text from {len(self.text_data)} pages")
        return self.text_data

    def extract_images(self, output_dir: str, dpi: int = 200) -> List[Dict]:
        """Extract images (page snapshots) from a PDF."""
        logger.info(f"Extracting images from {self.pdf_path}")

        POPPLER_PATH = r"C:\poppler\Library\bin"

        try:
            try:
                pages = convert_from_path(self.pdf_path, dpi=dpi)
            except Exception:
                logger.warning(f"Poppler not found in PATH, trying fallback path: {POPPLER_PATH}")
                pages = convert_from_path(self.pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)

            for page_num, page_img in enumerate(pages, 1):
                img_filename = f"page_{page_num:03d}.png"
                img_path = os.path.join(output_dir, img_filename)
                page_img.save(img_path, 'PNG')

                self.images.append({
                    'page_number': page_num,
                    'image_path': img_path,
                    'type': 'image'
                })

        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            raise

        logger.info(f"Extracted {len(self.images)} page images")
        return self.images

    def ocr_images(self) -> List[Dict]:
        """Run OCR on extracted images using Tesseract."""
        logger.info("Running OCR on extracted images")

        ocr_results = []
        for img_data in self.images:
            try:
                text = pytesseract.image_to_string(Image.open(img_data["image_path"]))
                cleaned_text = clean_text(text)
                if cleaned_text:
                    ocr_results.append({
                        'page_number': img_data["page_number"],
                        'content': cleaned_text,
                        'type': 'ocr_text'
                    })
            except Exception as e:
                logger.error(f"OCR failed for page {img_data['page_number']}: {e}")

        logger.info(f"OCR extracted text from {len(ocr_results)} images")
        return ocr_results

    def get_page_count(self) -> int:
        """Get total number of pages in the PDF."""
        with pdfplumber.open(self.pdf_path) as pdf:
            return len(pdf.pages)
