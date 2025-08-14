# modules/embeddings_image.py
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
from typing import List, Dict
from .utils import setup_logging

logger = setup_logging()

class ImageEmbeddingsGenerator:
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32'):
        logger.info(f"Loading image embedding model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def generate_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize the features (common practice with CLIP)
                embedding = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding = embedding.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for {image_path}: {e}")
            return np.array([])
    
    def process_images(self, image_data: List[Dict]) -> List[Dict]:
        """Process multiple images and add embeddings."""
        logger.info(f"Generating image embeddings for {len(image_data)} images")
        
        for img_data in image_data:
            if 'image_path' in img_data:
                embedding = self.generate_embedding(img_data['image_path'])
                if embedding.size > 0:
                    img_data['embedding'] = embedding.tolist()
                    img_data['embedding_dim'] = len(embedding)
                else:
                    logger.warning(f"Failed to generate embedding for {img_data['image_path']}")
        
        return image_data