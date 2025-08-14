# modules/ocr_engine.py
import pytesseract
from PIL import Image
import os
from typing import List, Dict
from .utils import setup_logging, clean_text, generate_id

logger = setup_logging()

class OCREngine:
    def __init__(self, confidence_threshold: int = 30):
        self.confidence_threshold = confidence_threshold
    
    def extract_text_from_image(self, image_path: str) -> Dict:
        """Extract text from a single image using OCR."""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Get OCR data with confidence scores
            ocr_data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT
            )
            
            # Filter text by confidence
            extracted_text = []
            for i in range(len(ocr_data['text'])):
                confidence = int(ocr_data['conf'][i])
                text = ocr_data['text'][i].strip()
                
                if confidence >= self.confidence_threshold and text:
                    extracted_text.append(text)
            
            # Join all text
            full_text = ' '.join(extracted_text)
            cleaned_text = clean_text(full_text)
            
            return {
                'image_path': image_path,
                'extracted_text': cleaned_text,
                'confidence_scores': [conf for conf in ocr_data['conf'] 
                                    if int(conf) >= self.confidence_threshold],
                'success': True
            }
            
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return {
                'image_path': image_path,
                'extracted_text': '',
                'confidence_scores': [],
                'success': False,
                'error': str(e)
            }
    
    def process_images(self, image_list: List[Dict]) -> List[Dict]:
        """Process multiple images with OCR."""
        logger.info(f"Starting OCR processing for {len(image_list)} images")
        
        ocr_results = []
        
        for img_data in image_list:
            logger.info(f"Processing OCR for page {img_data['page_number']}")
            
            result = self.extract_text_from_image(img_data['image_path'])
            
            ocr_result = {
                'id': generate_id(f"page_{img_data['page_number']}_ocr", "ocr"),
                'page_number': img_data['page_number'],
                'image_path': img_data['image_path'],
                'content': result['extracted_text'],
                'type': 'ocr',
                'confidence_scores': result.get('confidence_scores', []),
                'success': result['success']
            }
            
            if result.get('error'):
                ocr_result['error'] = result['error']
            
            ocr_results.append(ocr_result)
        
        successful_ocr = sum(1 for r in ocr_results if r['success'])
        logger.info(f"OCR completed: {successful_ocr}/{len(image_list)} successful")
        
        return ocr_results
