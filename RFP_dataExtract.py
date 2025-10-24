
import os
import json
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field, ValidationError
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import typing
from typing import get_args, get_origin

# --- LLM API ---
from google import genai
from google.genai import types

# --- RAG Dependencies ---
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# --- Define Pydantic Schema for RFP Data ---
class RFPData(BaseModel):
    Bid_Number: str
    Title: str
    Due_Date: str
    Bid_Submission_Type: str
    Term_of_Bid: str
    Pre_Bid_Meeting: str
    Installation: str
    Bid_Bond_Requirement: str
    Delivery_Date: str
    Payment_Terms: str
    Any_Additional_Documentation_Required: list[str]
    MFG_for_Registration: str
    Contract_or_Cooperative_to_use: str
    Model_no: str
    Part_no: str
    Product: str
    contact_info: str
    company_name: str
    Bid_Summary: str
    Product_Specification: list[str]

# --- Schema Conversion Function ---
def pydantic_to_gemini_schema(model: BaseModel) -> types.Schema:
    properties = {}
    required_fields = []

    TYPE_MAP = {
        str: types.Type.STRING,
        int: types.Type.INTEGER,
        float: types.Type.NUMBER,
        bool: types.Type.BOOLEAN,
    }

    for field_name, field in model.model_fields.items():
        field_type = field.annotation
        is_list = get_origin(field_type) in (list, typing.List)
        description = field.description or f"The {field_name} value."

        if is_list:
            inner_type = get_args(field_type)[0] if get_args(field_type) else str
            inner_gemini_type = TYPE_MAP.get(inner_type, types.Type.STRING)
            properties[field_name] = types.Schema(
                type=types.Type.ARRAY,
                description=description,
                items=types.Schema(type=inner_gemini_type)
            )
        else:
            type_enum = TYPE_MAP.get(field_type, types.Type.STRING)
            properties[field_name] = types.Schema(
                type=type_enum,
                description=description
            )
        if field.is_required:
            required_fields.append(field_name)

    return types.Schema(
        type=types.Type.OBJECT,
        properties=properties,
        required=required_fields
    )

GEMINI_SCHEMA_OBJECT = pydantic_to_gemini_schema(RFPData)

# --- Gemini Client ---
class GeminiClient:
    def __init__(self):
        if not os.environ.get("GEMINI_API_KEY"):
            raise EnvironmentError("GEMINI_API_KEY is not set in .env file.")
        self.client = genai.Client()

    def extract_structured_data(self, prompt: str, context_text: str):
        print("--- Calling Gemini 2.5 Pro with RAG context ---")
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GEMINI_SCHEMA_OBJECT
        )

        full_prompt = f"{prompt}\n\n--- RELEVANT CONTEXT ---\n{context_text}"

        response = self.client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[full_prompt],
            config=config,
        )
        return response.text

# --- Text Extraction Functions ---
def extract_text_from_html(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
        for element in soup(["script", "style", "nav", "footer"]):
            element.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        return '\n'.join(line for line in lines if line)
    except Exception as e:
        print(f"Error parsing HTML file {filepath}: {e}")
        return ""

def extract_text_from_pdf(filepath: str) -> str:
    text = ""
    try:
        doc = fitz.open(filepath)
        for page in doc:
            text += page.get_text()
        doc.close()
        return '\n'.join(line.strip() for line in text.splitlines() if line.strip())
    except Exception as e:
        print(f"Error parsing PDF file {filepath}: {e}")
        return ""

def extract_text_from_document(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    print(f"\nProcessing: {os.path.basename(filepath)} ({ext})")
    if ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif ext == '.html':
        return extract_text_from_html(filepath)
    else:
        print("Unsupported file type.")
        return ""

# --- RAG Setup ---
def initialize_chroma_collection():
    client = chromadb.Client()
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name="rfp_chunks", embedding_function=embedder)
    return collection

def chunk_text(text: str, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def add_to_chroma(collection, filepath: str, text: str):
    chunks = chunk_text(text)
    ids = [f"{os.path.basename(filepath)}_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=[{"file": filepath}] * len(chunks))
    print(f"Indexed {len(chunks)} chunks from {filepath} into Chroma.")
    return chunks

def retrieve_context(collection, filepath: str, top_k=3):
    query = f"Extract relevant details for {os.path.basename(filepath)} RFP"
    results = collection.query(query_texts=[query], n_results=top_k)
    contexts = results['documents'][0] if results and 'documents' in results else []
    return "\n\n".join(contexts)

# --- Process Function ---
def process_documents(document_files, llm_client, collection):
    all_results = {}
    prompt = (
        "You are an expert procurement analyst. "
        "Using the provided RFP context, fill in all fields accurately in the JSON schema. "
        "If data is missing, use 'Not Specified'."
    )

    for file in document_files:
        if not os.path.exists(file):
            print(f"Missing file: {file}")
            continue

        raw_text = extract_text_from_document(file)
        if not raw_text:
            print(f"No text extracted from {file}")
            continue

        add_to_chroma(collection, file, raw_text)
        context = retrieve_context(collection, file, top_k=5)

        try:
            response = llm_client.extract_structured_data(prompt, context)
            validated = RFPData.model_validate_json(response).model_dump()
            all_results[os.path.basename(file)] = validated
            print(f" Extracted structured data for {file}")
        except ValidationError as e:
            all_results[os.path.basename(file)] = {"status": "Validation Failed", "error": str(e)}
            print(f" Validation failed for {file}")
        except Exception as e:
            all_results[os.path.basename(file)] = {"status": "Extraction Failed", "error": str(e)}
            print(f" Extraction failed for {file}")

    return all_results

# --- Main Entry Point ---
if __name__ == "__main__":
    try:
        llm_client = GeminiClient()
    except Exception as e:
        print(f"Fatal Error: {e}")
        exit(1)

    document_files = [
        "Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf",
        "Student and Staff Computing Devices __SOURCING #168884__ - Bid Information - {3} _ BidNet Direct.html"
    ]

    collection = initialize_chroma_collection()
    results = process_documents(document_files, llm_client, collection)

    output_file = "rfp_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\n==========================")
    print(f"Extraction complete. Results saved to {output_file}")
