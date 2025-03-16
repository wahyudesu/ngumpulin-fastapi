import os
import requests
from typing import Union
from fastapi import FastAPI, HTTPException, Form
from supabase import create_client, Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from gliner import GLiNER
from dotenv import load_dotenv

# Load .env dari direktori saat ini
load_dotenv()

app = FastAPI()

# Get environment variables dengan default None dan raise error jika tidak ada
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize GLiNER with the base model
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
labels = ["Name", "ID"]

@app.get("/")
def read_root():
    return {"message": "Hello World"}

# menerima input id dan file url
@app.post("/upload")
async def upload_file(
    file_url: str = Form(...),
    folder: str = Form(...),
    Class: str = Form(...)
):
    try:
        # Load file PDF langsung dari URL
        loader = PyPDFLoader(file_url)
        documents = loader.load()

        # Split dokumen menjadi chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(documents)
        
        # Ambil chunk pertama
        first_chunk = chunks[0].page_content if chunks else ""

        # Ekstraksi entitas dari chunk pertama menggunakan GLiNER
        entities = model.predict_entities(first_chunk, labels)
        
        # Buat dictionary untuk menyimpan hasil ekstraksi
        extracted_data = {"Name": "", "ID": ""}
        for entity in entities:
            extracted_data[entity["label"]] = entity["text"]

        # Konversi semua chunks menjadi markdown content
        markdown_content = "\n\n".join(doc.page_content for doc in chunks)

        # Simpan ke Supabase dengan data entitas terpisah
        response = supabase.table("documents").insert({
            "nameStudent": extracted_data["Name"] or "admin",  # Gunakan "admin" jika Name kosong
            "grade": extracted_data["ID"],
            "class": Class,
            "documentUrl": file_url,
            "documentName": file_url,
            "folder": folder,
            "feedback": markdown_content
        }).execute()

        return {
            "message": "File processed and saved successfully to Supabase.",
            "extracted_entities": extracted_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")