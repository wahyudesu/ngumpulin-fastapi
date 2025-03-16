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
from pinecone import Pinecone
import re

load_dotenv()

app = FastAPI()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize GLiNER with the base model
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
labels = ["Name", "ID"]

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
embeddings = PineconeEmbeddings(model="multilingual-e5-large")

# homepage
@app.get("/")
def read_root():
    return {"message": "Hello World"}

# menerima input uuid dan file url
@app.post("/upload")
async def upload_file(
    uuid: str = Form(...),
    file_url: str = Form(...),
):
    try:
        # Load file PDF langsung dari URL
        loader = PyPDFLoader(file_url)
        documents = loader.load()

        # Hitung jumlah halaman
        page_count = len(documents)

        # Gabungkan semua konten halaman untuk menghitung jumlah kalimat
        full_text = " ".join(doc.page_content for doc in documents)
        # Pisahkan kalimat berdasarkan tanda baca (. ! ?), lalu filter yang tidak kosong
        sentences = [s.strip() for s in re.split(r'[.!?]+', full_text) if s.strip()]
        sentence_count = len(sentences)

        # Split dokumen menjadi chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(documents)
        
        # Ambil chunk pertama dan ekstraksi entity 
        first_chunk = chunks[0].page_content if chunks else ""
        entities = model.predict_entities(first_chunk, labels)
        
        # Buat dictionary untuk menyimpan hasil ekstraksi
        extracted_data = {"Name": "", "ID": ""}
        for entity in entities:
            extracted_data[entity["label"]] = entity["text"]

        # Konversi semua chunks menjadi markdown content
        markdown_content = "\n\n".join(doc.page_content for doc in chunks)

        # Konversi markdown content ke vektor menggunakan embeddings
        vector = embeddings.embed_documents([markdown_content])[0]  # embed_documents mengembalikan list, ambil elemen pertama

        # Update baris di Supabase berdasarkan uuid
        response = supabase.table("documents").update({
            "nameStudent": extracted_data["Name"] or "null",
            "NRP": extracted_data["ID"],
            "isiTugas": markdown_content,
            "embedding": vector,
            "page": page_count,       # Simpan jumlah halaman
            "sentences": sentence_count  # Simpan jumlah kalimat
        }).eq("id", uuid).execute()

        # Periksa apakah ada baris yang diupdate
        if not response.data:
            raise HTTPException(status_code=404, detail=f"No record found with uuid: {uuid}")

        return {
            "message": "File processed and record updated successfully in Supabase.",
            "extracted_entities": extracted_data,
            "vector": vector,
            "page_count": page_count,
            "sentence_count": sentence_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")