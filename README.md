📚 RAG with Local LLM (Ollama)
This project implements a Retrieval-Augmented Generation (RAG) pipeline using a local LLM powered by Ollama. It allows you to upload PDFs, extract and vectorize content, and query information using natural language, all locally for maximum privacy and offline capability.

✅ Features
Local LLM integration with Ollama (privacy-first, no cloud API).

RAG architecture for improved context-based answers.

PDF ingestion and processing (extract text, chunk, vectorize).

Embeddings using Ollama.

Vector database for semantic search.

Streamlit interface for easy interaction.

🛠️ Requirements
Python 3.9+

Ollama installed (Download here)

Dependencies (install via requirements.txt):

bash
Copy
Edit
pip install -r requirements.txt
📂 Project Structure
graphql
Copy
Edit
.
├── local_llm/
│   ├── crawl.py           # Upload, extract, and vectorize PDFs (Ollama embeddings)
│   ├── model.py           # LLM integration, model setup, RAG templates
│   ├── pdfinjest.py       # Main script: Streamlit app, PDF ingestion, retrieval
│   ├── vectordatabase.py  # Vector database management functions
├── requirements.txt
├── README.md
🚀 Getting Started
1. Install Ollama
Download and install Ollama for your platform:
Ollama Installation Guide

Verify:

bash
Copy
Edit
ollama --version
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Pull Your Model
For example, to use LLaMA 2:

bash
Copy
Edit
ollama pull llama2
Other models available: mistral, gemma, etc. (see Ollama Library).

4. Run the Streamlit App
bash
Copy
Edit
streamlit run local_llm/pdfinjest.py
This will:

Start the UI for uploading PDFs.

Ingest, process, and store embeddings in the vector database.

Enable querying the uploaded documents with local LLM.

🔍 Workflow
Upload PDF → crawl.py extracts text, chunks it, and creates embeddings using Ollama.

Store in Vector DB → vectordatabase.py handles insertion and retrieval.

Query with RAG → model.py integrates the LLM and uses retrieved context for responses.

Interact via Streamlit → pdfinjest.py provides a user-friendly interface.

🖥️ Example Usage
Upload and Process PDF
bash
Copy
Edit
python local_llm/crawl.py --file sample.pdf
Query Through UI
Launch Streamlit and ask:

sql
Copy
Edit
What are the key points discussed in the PDF?
✅ Advantages
100% local execution (privacy-first).

Offline capable.

Scalable to multiple PDFs and models.

⚠️ Limitations
Requires sufficient RAM & storage for large models.

Performance depends on model size and hardware.

📜 License
MIT License.

