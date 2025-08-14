# modules/embeddings_text.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
from .utils import setup_logging

logger = setup_logging()

class TextEmbeddingsGenerator:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        logger.info(f"Loading text embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_tensor=False
        )
        
        return np.array(embeddings)
    
    def process_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Process chunks and add embeddings."""
        if not chunks:
            return []
        
        # Extract texts
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
            chunk['embedding_dim'] = self.embedding_dim
        
        return chunks
