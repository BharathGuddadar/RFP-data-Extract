Sure thing, Bharath ✅ — here’s a clean, professional **`README.md`** written in the same concise format and matching your updated script (`rfp_extractor_rag.py`):

---

````markdown
# 🧠 RFP Data Extraction Program (`rfp_extractor_rag.py`)

This project implements a **Retrieval-Augmented Generation (RAG)** system to extract **structured data** from **RFP (Request for Proposal)** documents — supporting both **PDF** and **HTML** formats.

It combines **semantic retrieval (ChromaDB)** with **LLM-based structured extraction (Gemini 2.5-Pro)** to generate validated **JSON outputs** that match a predefined schema for consistent and accurate data extraction.

---

## 🚀 Features

- Extracts data fields like Bid Number, Title, Due Date, Term, Warranty, and Delivery Dates.  
- Uses **Gemini 2.5-Pro** for JSON-structured extraction aligned with a **Pydantic schema**.  
- Employs **ChromaDB** + **SentenceTransformers** (`all-MiniLM-L6-v2`) for contextual retrieval.  
- Supports both **PDF (PyMuPDF)** and **HTML (BeautifulSoup)** documents.  
- Logs all processing with validation and error handling.  
- Generates a final `rfp_data.json` output file with structured results.

---

## 🧩 Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
````

### requirements.txt

```
chromadb
sentence-transformers
google-genai
pydantic
beautifulsoup4
pymupdf
python-dotenv
```

---

## ⚙️ Environment Setup

Create a `.env` file in your project root directory with your Gemini API key:

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

---

## 🏃‍♂️ How to Run

1. Place your **RFP files** (`.pdf` or `.html`) in the project directory.

2. Run the script using:

   ```bash
   python rfp_extractor_rag.py
   ```

   Or specify input files explicitly:

   ```bash
   python rfp_extractor_rag.py --input "Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf" "Student and Staff Computing Devices __SOURCING #168884__ - Bid Information - {3} _ BidNet Direct.html"
   ```

3. The extracted structured data will be saved as:

   ```
   rfp_data.json
   ```

---

## 📂 Output Example

```json
{
  "Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf": {
    "Bid_Number": "JA-207652",
    "Title": "Student and Staff Computing Devices",
    "Delivery_Date": "Starting in September 2024",
    "Warranty": "Minimum one-year OEM for Chromebooks, three-year OEM for Windows laptops"
  }
}
```

---

## 🧠 Models and Tools Used

| Component             | Purpose                       | Model / Library        |
| --------------------- | ----------------------------- | ---------------------- |
| **LLM (Extraction)**  | Structured JSON extraction    | Gemini 2.5-Pro         |
| **Embedding Model**   | Text embeddings for retrieval | all-MiniLM-L6-v2       |
| **Vector Store**      | Local semantic storage        | ChromaDB               |
| **Schema Validation** | Output consistency            | Pydantic               |
| **Text Extraction**   | Read content from files       | PyMuPDF, BeautifulSoup |

---

## 📦 Output Files

| File               | Description                       |
| ------------------ | --------------------------------- |
| `rfp_data.json`    | Final structured output           |
| `chroma_db/`       | Persistent local ChromaDB storage |
| `.env`             | Contains Gemini API key           |
| `requirements.txt` | Dependencies list                 |

---

## 🧑‍💻 Author

**Bharath G P**
AI/ML Engineer | RAG & LLM Applications | Cloud & Data Extraction Systems

```

---

