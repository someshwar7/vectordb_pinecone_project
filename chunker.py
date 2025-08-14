from typing import List, Dict
from .utils import setup_logging, generate_id

logger = setup_logging()

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into overlapping chunks."""
        if not text or len(text.strip()) == 0:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk_data = {
                'content': chunk_text,
                'chunk_index': len(chunks),
                'word_count': len(chunk_words),
                'char_count': len(chunk_text)
            }
            
            # Add metadata if provided
            if metadata:
                chunk_data.update(metadata)
            
            # Generate unique ID
            chunk_data['id'] = generate_id(
                f"{metadata.get('page_number', 'unknown')}_{chunk_data['chunk_index']}_{chunk_text[:50]}",
                "chunk"
            )
            
            chunks.append(chunk_data)
        
        return chunks
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process multiple documents into chunks."""
        logger.info(f"Chunking {len(documents)} documents")
        
        all_chunks = []
        
        for doc in documents:
            if not doc.get('content'):
                continue
            
            metadata = {
                'page_number': doc.get('page_number'),
                'type': doc.get('type', 'unknown'),
                'source_id': doc.get('id')
            }
            
            chunks = self.chunk_text(doc['content'], metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks")
        return all_chunks