import os
import sys
from config.settings import Config
from modules.pdf_extractor import PDFExtractor
from modules.ocr_engine import OCREngine
from modules.chunker import TextChunker
from modules.embeddings_text import TextEmbeddingsGenerator
from modules.embeddings_image import ImageEmbeddingsGenerator
from modules.pinecone_client import PineconeClient
from modules.utils import setup_logging, save_jsonl, load_jsonl
from tqdm import tqdm

def main():
    # Setup
    logger = setup_logging()
    
    # Validate configuration and create directories
    try:
        Config.validate_config()
        Config.create_directories()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Please check your .env file and ensure PINECONE_API_KEY is set")
        return 1
    
    # Check for PDF file
    pdf_path = os.path.join(Config.RAW_DIR, 'book.pdf')
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at {pdf_path}")
        logger.info("Please place your PDF file at data/raw/book.pdf")
        return 1
    
    logger.info("Starting PDF processing pipeline...")
    logger.info(f"Processing file: {pdf_path}")
    logger.info(f"Output directory: {Config.PROCESSED_DIR}")
    
    try:
        # Step 1: Extract text and images from PDF
        logger.info("=" * 50)
        logger.info("STEP 1: PDF TEXT & IMAGE EXTRACTION")
        logger.info("=" * 50)
        
        extractor = PDFExtractor(pdf_path)
        
        # Extract text
        logger.info("Extracting text from PDF...")
        text_data = extractor.extract_text()
        text_file = os.path.join(Config.PROCESSED_DIR, 'pdf_text.jsonl')
        save_jsonl(text_data, text_file)
        logger.info(f"Extracted text from {len(text_data)} pages, saved to {text_file}")
        
        # Extract images
        logger.info("Extracting images from PDF...")
        image_data = extractor.extract_images(Config.IMAGES_DIR)
        images_file = os.path.join(Config.PROCESSED_DIR, 'pdf_images.jsonl')
        save_jsonl(image_data, images_file)
        logger.info(f"Extracted {len(image_data)} images, saved to {images_file}")
        
        # Step 2: OCR Processing
        logger.info("=" * 50)
        logger.info("STEP 2: OCR PROCESSING")
        logger.info("=" * 50)
        
        if image_data:
            ocr_engine = OCREngine(Config.OCR_CONFIDENCE_THRESHOLD)
            logger.info(f"Processing {len(image_data)} images with OCR...")
            ocr_results = ocr_engine.process_images(image_data)
            ocr_file = os.path.join(Config.PROCESSED_DIR, 'ocr_text.jsonl')
            save_jsonl(ocr_results, ocr_file)
            
            successful_ocr = [r for r in ocr_results if r.get('success', False) and r.get('content')]
            logger.info(f"OCR completed: {len(successful_ocr)}/{len(ocr_results)} successful extractions")
        else:
            logger.info("No images found, skipping OCR step")
            ocr_results = []
        
        # Step 3: Merge and chunk text
        logger.info("=" * 50)
        logger.info("STEP 3: TEXT CHUNKING")
        logger.info("=" * 50)
        
        # Combine text data and OCR results
        all_documents = text_data.copy()
        
        # Add successful OCR results
        for ocr in ocr_results:
            if ocr.get('success', False) and ocr.get('content'):
                all_documents.append({
                    'page_number': ocr['page_number'],
                    'content': ocr['content'],
                    'type': 'ocr',
                    'id': ocr['id']
                })
        
        logger.info(f"Total documents to chunk: {len(all_documents)}")
        
        chunker = TextChunker(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        chunks = chunker.process_documents(all_documents)
        chunks_file = os.path.join(Config.PROCESSED_DIR, 'text_chunks.jsonl')
        save_jsonl(chunks, chunks_file)
        logger.info(f"Created {len(chunks)} text chunks, saved to {chunks_file}")
        
        # Step 4: Generate text embeddings
        logger.info("=" * 50)
        logger.info("STEP 4: TEXT EMBEDDINGS GENERATION")
        logger.info("=" * 50)
        
        logger.info(f"Generating embeddings using model: {Config.TEXT_EMBEDDING_MODEL}")
        text_embedder = TextEmbeddingsGenerator(Config.TEXT_EMBEDDING_MODEL)
        chunks_with_embeddings = text_embedder.process_chunks(chunks)
        
        embeddings_file = os.path.join(Config.PROCESSED_DIR, 'chunks_with_embeddings.jsonl')
        save_jsonl(chunks_with_embeddings, embeddings_file)
        logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
        
        # Step 5: Generate image embeddings (optional)
        logger.info("=" * 50)
        logger.info("STEP 5: IMAGE EMBEDDINGS GENERATION")
        logger.info("=" * 50)
        
        images_with_embeddings = []
        if image_data:
            try:
                logger.info(f"Generating image embeddings using model: {Config.IMAGE_EMBEDDING_MODEL}")
                image_embedder = ImageEmbeddingsGenerator(Config.IMAGE_EMBEDDING_MODEL)
                images_with_embeddings = image_embedder.process_images(image_data.copy())
                
                images_embeddings_file = os.path.join(Config.PROCESSED_DIR, 'images_with_embeddings.jsonl')
                save_jsonl(images_with_embeddings, images_embeddings_file)
                logger.info(f"Generated embeddings for {len(images_with_embeddings)} images")
            except Exception as e:
                logger.warning(f"Image embeddings generation failed: {e}")
                logger.info("Continuing with text embeddings only...")
        else:
            logger.info("No images to process, skipping image embeddings")
        
        # Step 6: Upload to Pinecone
        logger.info("=" * 50)
        logger.info("STEP 6: PINECONE UPLOAD")
        logger.info("=" * 50)
        
        # Initialize Pinecone client using config
        logger.info("Initializing Pinecone client...")
        pinecone_client = PineconeClient.from_config()
        
        # Create index with appropriate dimension
        embedding_dim = getattr(text_embedder, 'embedding_dim', 384)  # Default for all-MiniLM-L6-v2
        logger.info(f"Creating/connecting to index with dimension: {embedding_dim}")
        pinecone_client.create_index(dimension=embedding_dim)
        
        # Prepare all embeddings for upload
        all_embeddings = []
        
        # Add text embeddings
        text_embeddings_count = 0
        for chunk in chunks_with_embeddings:
            if 'embedding' in chunk and chunk['embedding']:
                all_embeddings.append(chunk)
                text_embeddings_count += 1
        
        # Add image embeddings if available
        image_embeddings_count = 0
        for img in images_with_embeddings:
            if 'embedding' in img and img['embedding']:
                # Ensure image data has required fields
                img_embedding = {
                    'id': f"img_page_{img.get('page_number', 'unknown')}_{img.get('id', 'unknown')}",
                    'embedding': img['embedding'],
                    'content': f"Image from page {img.get('page_number', 'unknown')}",
                    'type': 'image',
                    'page_number': img.get('page_number'),
                    'image_path': img.get('image_path'),
                    'source': 'book.pdf'
                }
                all_embeddings.append(img_embedding)
                image_embeddings_count += 1
        
        logger.info(f"Prepared {len(all_embeddings)} embeddings for upload:")
        logger.info(f"  - Text embeddings: {text_embeddings_count}")
        logger.info(f"  - Image embeddings: {image_embeddings_count}")
        
        # Upload to Pinecone
        if all_embeddings:
            logger.info("Uploading embeddings to Pinecone...")
            uploaded_count = pinecone_client.upsert_embeddings(all_embeddings, batch_size=50)
            logger.info(f"Successfully uploaded {uploaded_count} embeddings")
            
            # Get index stats
            stats = pinecone_client.get_index_stats()
            if stats:
                logger.info(f"Index statistics: {stats}")
        else:
            logger.warning("No embeddings to upload")
            return 1
        
        # Step 7: Test search
        logger.info("=" * 50)
        logger.info("STEP 7: TEST SEARCH")
        logger.info("=" * 50)
        
        test_queries = [
            "What is the main topic of this book?",
            "Tell me about the characters",
            "What happens in the story?",
            "Summary of key concepts"
        ]
        
        logger.info("Running test queries...")
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n[Query {i}] '{query}'")
            
            try:
                # Generate query embedding
                query_embedding = text_embedder.model.encode([query])[0].tolist()
                
                # Search
                results = pinecone_client.query(query_embedding, top_k=3)
                
                matches = results.get('matches', [])
                if matches:
                    logger.info(f"Found {len(matches)} results:")
                    for j, match in enumerate(matches, 1):
                        score = match.get('score', 0)
                        metadata = match.get('metadata', {})
                        content_preview = str(metadata.get('content', ''))[:150] + "..."
                        doc_type = metadata.get('type', 'unknown')
                        page_num = metadata.get('page_number', 'unknown')
                        
                        logger.info(f"  {j}. Score: {score:.4f} | Type: {doc_type} | Page: {page_num}")
                        logger.info(f"     Preview: {content_preview}")
                else:
                    logger.info("No matches found")
                    
            except Exception as e:
                logger.error(f"Error testing query '{query}': {e}")
        
        # Final summary
        logger.info("\n" + "=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info("SUMMARY:")
        logger.info(f"  📄 PDF pages processed: {len(text_data)}")
        logger.info(f"  🖼️  Images extracted: {len(image_data)}")
        logger.info(f"  📝 OCR extractions: {len([r for r in ocr_results if r.get('success')])}")
        logger.info(f"  📚 Text chunks created: {len(chunks)}")
        logger.info(f"  🔢 Text embeddings: {text_embeddings_count}")
        logger.info(f"  🎨 Image embeddings: {image_embeddings_count}")
        logger.info(f"  ☁️  Total uploaded to Pinecone: {len(all_embeddings)}")
        logger.info(f"  🗂️  Pinecone index: {Config.PINECONE_INDEX_NAME}")
        logger.info(f"  📁 Processed files saved to: {Config.PROCESSED_DIR}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error("Check the logs above for detailed error information")
        raise

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)