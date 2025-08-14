# modules/pinecone_client.py
from pinecone import Pinecone  # Updated import
from typing import List, Dict, Optional
import time
from .utils import setup_logging
from config.settings import Config

logger = setup_logging()

class PineconeClient:
    def __init__(self, api_key: str = None, index_name: str = None):
        """
        Initialize PineconeClient. Uses config defaults if parameters not provided.
        
        Args:
            api_key: Pinecone API key (defaults to Config.PINECONE_API_KEY)
            index_name: Index name (defaults to Config.PINECONE_INDEX_NAME)
        """
        self.api_key = api_key or Config.PINECONE_API_KEY
        self.index_name = index_name or Config.PINECONE_INDEX_NAME
        self.pc = None  # Pinecone client instance
        self.index = None
        
        # Validate configuration
        if not self.api_key:
            raise ValueError("Pinecone API key is required. Check your .env file.")
        
        self.initialize()
    
    @classmethod
    def from_config(cls):
        """Create PineconeClient instance using configuration settings."""
        Config.validate_config()
        return cls()
    
    def initialize(self):
        """Initialize Pinecone connection."""
        logger.info("Initializing Pinecone connection")
        
        try:
            # Initialize Pinecone client (new syntax)
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check if index exists before trying to connect
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                logger.warning(f"Index '{self.index_name}' does not exist. Call create_index() first.")
            else:
                self.index = self.pc.Index(self.index_name)
                logger.info(f"Connected to existing index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise
    
    def create_index(self, dimension: int = 384, metric: str = 'cosine'):
        """
        Create Pinecone index if it doesn't exist.
        
        Args:
            dimension: Vector dimension (default 384 for all-MiniLM-L6-v2)
            metric: Distance metric ('cosine', 'euclidean', 'dotproduct')
        """
        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name} (dimension: {dimension}, metric: {metric})")
                
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric,
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                )
                
                # Wait for index to be ready
                logger.info("Waiting for index to be ready...")
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                
                logger.info(f"Index {self.index_name} created successfully")
            else:
                logger.info(f"Index {self.index_name} already exists")
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def clean_metadata_value(self, value):
        """Clean a single metadata value to ensure Pinecone compatibility."""
        if value is None or value == "null":
            return ""  # Replace null with empty string
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, list):
            # Ensure all list items are strings and not null
            return [str(item) for item in value if item is not None and item != "null"]
        else:
            # Convert other types to string
            return str(value)
    
    def clean_metadata(self, metadata: Dict) -> Dict:
        """Clean metadata to ensure Pinecone compatibility."""
        if not isinstance(metadata, dict):
            return {}
        
        cleaned = {}
        for key, value in metadata.items():
            cleaned_value = self.clean_metadata_value(value)
            # Only add non-empty values
            if cleaned_value or cleaned_value == 0 or cleaned_value is False:
                cleaned[key] = cleaned_value
        
        return cleaned
    
    def upsert_embeddings(self, embeddings_data: List[Dict], batch_size: int = 100):
        """
        Upsert embeddings to Pinecone with metadata cleaning.
        
        Args:
            embeddings_data: List of dicts with 'id', 'embedding', and metadata
            batch_size: Number of vectors to upsert per batch
        """
        if not self.index:
            raise ValueError("Index not initialized. Call create_index() first.")
            
        logger.info(f"Upserting {len(embeddings_data)} embeddings to Pinecone")
        
        successful_upserts = 0
        
        for i in range(0, len(embeddings_data), batch_size):
            batch = embeddings_data[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(embeddings_data)-1)//batch_size + 1
            
            vectors = []
            for item in batch:
                if 'embedding' not in item or not item['embedding']:
                    logger.warning(f"Skipping item {item.get('id', 'unknown')} - no embedding")
                    continue
                
                # Prepare metadata (remove embedding from metadata)
                raw_metadata = {k: v for k, v in item.items() 
                               if k not in ['embedding', 'embedding_dim']}
                
                # Clean metadata to remove null values and ensure compatibility
                metadata = self.clean_metadata(raw_metadata)
                
                # Ensure we have a valid source_id
                if not metadata.get('source_id'):
                    metadata['source_id'] = item.get('id', f"unknown_{i}")
                
                # Ensure content is string and not too long for metadata
                if 'content' in metadata:
                    content_str = str(metadata['content'])
                    if len(content_str) > 40000:  # Pinecone metadata limit
                        metadata['content'] = content_str[:40000] + "..."
                
                vectors.append({
                    'id': str(item['id']),  # Ensure ID is string
                    'values': item['embedding'],
                    'metadata': metadata
                })
            
            if vectors:
                try:
                    self.index.upsert(vectors=vectors)
                    successful_upserts += len(vectors)
                    logger.info(f"Upserted batch {batch_num}/{total_batches} ({len(vectors)} vectors)")
                except Exception as e:
                    logger.error(f"Error upserting batch {batch_num}: {e}")
                    logger.error(f"First vector dimension: {len(vectors[0]['values']) if vectors else 'N/A'}")
                    # Log problematic metadata for debugging
                    if vectors:
                        logger.error(f"Sample metadata: {vectors[0]['metadata']}")
                    raise
        
        logger.info(f"Successfully upserted {successful_upserts} vectors to Pinecone")
        return successful_upserts
    
    def query(self, query_embedding: List[float], top_k: int = 5, 
              filter_dict: Optional[Dict] = None, include_values: bool = False) -> Dict:
        """
        Query Pinecone index.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Metadata filter
            include_values: Whether to include vector values in results
            
        Returns:
            Query results from Pinecone
        """
        if not self.index:
            raise ValueError("Index not initialized. Call create_index() first.")
            
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True,
                include_values=include_values
            )
            
            logger.info(f"Query returned {len(results.get('matches', []))} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            raise
    
    def get_index_stats(self):
        """Get index statistics."""
        if not self.index:
            logger.warning("Index not initialized")
            return None
            
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return None
    
    def delete_vectors(self, ids: List[str]):
        """Delete vectors by IDs."""
        if not self.index:
            raise ValueError("Index not initialized")
            
        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors")
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise
    
    def delete_all(self, confirm: bool = False):
        """Delete all vectors from the index."""
        if not confirm:
            logger.warning("delete_all() requires confirm=True parameter")
            return
            
        if not self.index:
            raise ValueError("Index not initialized")
            
        try:
            self.index.delete(delete_all=True)
            logger.info("Deleted all vectors from index")
        except Exception as e:
            logger.error(f"Error deleting all vectors: {e}")
            raise