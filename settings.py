import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    # Removed PINECONE_ENVIRONMENT - not needed for Pinecone v2.2.4
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'book-embeddings')
    
    # Model Configuration
    TEXT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    IMAGE_EMBEDDING_MODEL = 'openai/clip-vit-base-patch32'
    
    # Processing Configuration
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    OCR_CONFIDENCE_THRESHOLD = 30
    
    # Paths (relative to project root)
    DATA_DIR = 'data'
    RAW_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    TEMP_DIR = os.path.join(DATA_DIR, 'temp')
    IMAGES_DIR = os.path.join(TEMP_DIR, 'page_images')
    EMBEDDINGS_DIR = os.path.join(PROCESSED_DIR, 'embeddings')
    
    # File paths for processed data
    TEXT_CHUNKS_FILE = os.path.join(PROCESSED_DIR, 'text_chunks.jsonl')
    OCR_TEXT_FILE = os.path.join(PROCESSED_DIR, 'ocr_text.jsonl')
    
    # Ensure all directories exist
    @classmethod
    def create_directories(cls):
        """Create all necessary directories if they don't exist."""
        dirs = [
            cls.RAW_DIR, 
            cls.PROCESSED_DIR, 
            cls.TEMP_DIR,
            cls.IMAGES_DIR, 
            cls.EMBEDDINGS_DIR
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    # Validation method
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present."""
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        return True