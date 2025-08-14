# README.md
# 📜 Pinecone Vector DB with OCR for Text & Images

A complete Python 3.11 pipeline that processes PDF documents (with both text and images) into a searchable Pinecone vector database using OCR, text chunking, and embeddings.

## 🚀 Features

- **PDF Text Extraction**: Extract text directly from PDF pages
- **Image Extraction & OCR**: Extract images and run OCR to capture embedded text
- **Smart Text Chunking**: Break text into meaningful chunks with overlap
- **Dual Embeddings**: Generate embeddings for both text and images
- **Pinecone Integration**: Store and search embeddings in Pinecone vector DB
- **End-to-End Pipeline**: Automated processing from PDF to searchable database

## 📁 Project Structure

```
vector_db_project/
├── app.py                               # Main pipeline runner
├── search_example.py                    # Example search script
├── requirements.txt                     # Python dependencies
├── .env                                # Environment variables (create this)
│
├── config/
│   └── settings.py                      # Configuration settings
│
├── data/
│   ├── raw/
│   │   └── book.pdf                     # Place your PDF here
│   ├── processed/
│   │   ├── text_chunks.jsonl           # Chunked text data
│   │   ├── ocr_text.jsonl              # OCR-extracted text
│   │   └── embeddings/                 # Stored embedding files
│   └── temp/
│       └── page_images/                # Extracted images from PDF
│
├── modules/
│   ├── pdf_extractor.py               # PDF text/image extraction
│   ├── ocr_engine.py                  # OCR processing
│   ├── chunker.py                     # Text chunking logic
│   ├── embeddings_text.py             # Text embeddings generation
│   ├── embeddings_image.py            # Image embeddings (CLIP)
│   ├── pinecone_client.py             # Pinecone operations
│   └── utils.py                       # Helper functions
│
├── notebooks/
│   └── testing_pipeline.ipynb         # Testing and debugging notebook
│
└── README.md
```

## 🛠️ Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. System Requirements

**For OCR (Tesseract):**
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: Download from [GitHub releases](https://github.com/tesseract-ocr/tesseract)

**For PDF to Image conversion:**
- Ubuntu/Debian: `sudo apt-get install poppler-utils`
- macOS: `brew install poppler`
- Windows: Download from [poppler releases](http://blog.alivate.com.au/poppler-windows/)

### 3. Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp-free
PINECONE_INDEX_NAME=book-embeddings
```

### 4. Add Your PDF

Place your PDF file at `data/raw/book.pdf`

## 🚀 Usage

### Run the Complete Pipeline

```bash
python app.py
```

This will:
1. Extract text and images from your PDF
2. Run OCR on all images to extract embedded text
3. Chunk the combined text into manageable pieces
4. Generate embeddings for text chunks and images
5. Upload everything to Pinecone
6. Run test searches to verify functionality

### Search the Database

After the pipeline completes, use the search script:

```bash
python search_example.py
```

Or search programmatically:

```python
from config.settings import Config
from modules.embeddings_text import TextEmbeddingsGenerator
from modules.pinecone_client import PineconeClient

# Initialize components
text_embedder = TextEmbeddingsGenerator(Config.TEXT_EMBEDDING_MODEL)
pinecone_client = PineconeClient(
    Config.PINECONE_API_KEY,
    Config.PINECONE_ENVIRONMENT, 
    Config.PINECONE_INDEX_NAME
)

# Search
query = "What is the main topic?"
query_embedding = text_embedder.model.encode([query])[0].tolist()
results = pinecone_client.query(query_embedding, top_k=5)
```

### Debug with Jupyter Notebook

Use the testing notebook to debug individual components:

```bash
jupyter notebook notebooks/testing_pipeline.ipynb
```

## ⚙️ Configuration

Modify `config/settings.py` to customize:

- **Chunk size and overlap**: `CHUNK_SIZE = 500`, `CHUNK_OVERLAP = 50`
- **OCR confidence threshold**: `OCR_CONFIDENCE_THRESHOLD = 30`
- **Embedding models**: Change `TEXT_EMBEDDING_MODEL` or `IMAGE_EMBEDDING_MODEL`
- **Pinecone settings**: Index name, environment, etc.

## 📊 Pipeline Output

The pipeline generates several intermediate files in `data/processed/`:

- `pdf_text.jsonl`: Raw text extracted from PDF
- `pdf_images.jsonl`: Metadata for extracted images  
- `ocr_text.jsonl`: Text extracted from images via OCR
- `text_chunks.jsonl`: Chunked text data
- `chunks_with_embeddings.jsonl`: Text chunks with embeddings
- `images_with_embeddings.jsonl`: Images with CLIP embeddings

## 🔍 Search Features

The system supports:

- **Semantic text search**: Find content by meaning, not just keywords
- **Cross-modal search**: Search images using text queries (via CLIP)
- **Filtered search**: Search specific pages or content types
- **Similarity scoring**: Results ranked by relevance

## 🐛 Troubleshooting

**Common Issues:**

1. **Tesseract not found**: Ensure Tesseract is installed and in PATH
2. **Poppler not found**: Install poppler-utils for PDF to image conversion
3. **CUDA out of memory**: Use CPU-only versions of models if GPU memory is limited
4. **Pinecone connection failed**: Check API key and environment settings
5. **No text extracted**: Check if PDF is text-based or image-based

**Debug Steps:**

1. Use the Jupyter notebook to test individual components
2. Check log output for specific error messages  
3. Verify your PDF file is readable and not corrupted
4. Test with a smaller PDF first

## 🔧 Customization

**Add new embedding models:**

```python
# In embeddings_text.py
from sentence_transformers import SentenceTransformer

class TextEmbeddingsGenerator:
    def __init__(self, model_name='your-model-name'):
        self.model = SentenceTransformer(model_name)
```

**Modify chunking strategy:**

```python
# In chunker.py
class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        # Your custom chunking logic
```

**Add preprocessing steps:**

```python
# In utils.py
def clean_text(text: str) -> str:
    # Add your custom text cleaning logic
```

## 📝 License

This project is open source. Feel free to modify and distribute as needed.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📚 Dependencies

- **pdfplumber**: PDF text extraction
- **pdf2image**: Convert PDF pages to images
- **pytesseract**: OCR processing
- **sentence-transformers**: Text embeddings
- **transformers**: CLIP image embeddings  
- **pinecone-client**: Vector database operations
- **torch/torchvision**: Deep learning models
- **Pillow**: Image processing

## 🎯 Use Cases

Perfect for:
- **Educational content**: Process textbooks, study materials
- **Research papers**: Make academic papers searchable
- **Technical documentation**: Index manuals, guides
- **Legal documents**: Search contracts, policies
- **Historical archives**: Digitize and search old documents

---

Ready to transform your PDFs into searchable knowledge bases! 🚀
