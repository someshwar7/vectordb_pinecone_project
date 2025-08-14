# modules/utils.py
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def save_jsonl(data: List[Dict], filepath: str):
    """Save data as JSONL format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSONL format."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep punctuation
    text = text.replace('\x00', '')
    text = text.replace('\ufffd', '')
    
    return text.strip()

def generate_id(content: str, prefix: str = "") -> str:
    """Generate unique ID from content."""
    hash_obj = hashlib.md5(content.encode('utf-8'))
    return f"{prefix}_{hash_obj.hexdigest()[:8]}" if prefix else hash_obj.hexdigest()[:8]

def save_embeddings(embeddings: np.ndarray, filepath: str):
    """Save embeddings to file."""
    np.save(filepath, embeddings)

def load_embeddings(filepath: str) -> np.ndarray:
    """Load embeddings from file."""
    return np.load(filepath)