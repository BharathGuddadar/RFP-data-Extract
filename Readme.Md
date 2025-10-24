```
## README: RFP Data Extraction Program

This project implements a Retrieval-Augmented Generation (RAG) system to extract structured data from RFP (Request for Proposal) documents — including both PDF and HTML formats.

It combines information retrieval (ChromaDB) and LLM-based structured extraction (Gemini 2.5-Pro) to produce validated JSON outputs aligned with a predefined schema.


-----

Dependencies

Install via pip install -r requirements.txt

requirements.txt

chromadb
sentence-transformers
google-genai
pydantic
beautifulsoup4
pymupdf
python-dotenv


Environment Setup

Create a .env file in the project root:

GEMINI_API_KEY=your_google_gemini_api_key_here

##How to Run

Place your RFP files (PDF or HTML) in the same directory (or update file paths).

Run the script:

python rag_v2.py

```