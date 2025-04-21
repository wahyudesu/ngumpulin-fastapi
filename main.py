import os
import re
import json
import numpy as np
from typing import Union
from fastapi import FastAPI, HTTPException, Form
from supabase import create_client, Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeEmbeddings
from gliner import GLiNER
from dotenv import load_dotenv
from pinecone import Pinecone
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from agent_feedback import process_assignment

import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

load_dotenv()

app = FastAPI()

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
labels = ["Name", "ID"]
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
embeddings = PineconeEmbeddings(model="multilingual-e5-large")

# Load the saved model
with open('best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Main route
@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Upload route
@app.post("/upload")
async def upload_file(uuid: str = Form(...), file_url: str = Form(...)):
    try:
        # Parse pdf
        loader = PyPDFLoader(file_url)
        documents = loader.load()
        page_count = len(documents)
        full_text = " ".join(doc.page_content for doc in documents)
        sentence_count = len([s for s in re.split(r'[.!?]+', full_text) if s.strip()])

        # chunking and embedding to vector DB
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(documents)
        markdown_content = "\n\n".join(doc.page_content for doc in chunks)
        vector = embeddings.embed_documents([markdown_content])[0]

        # Entity extraction
        first_chunk = chunks[0].page_content if chunks else ""
        entities = model.predict_entities(first_chunk, labels)
        extracted_data = {"Name": "", "ID": ""}
        for entity in entities:
            extracted_data[entity["label"]] = entity["text"]

        # Get data specific documents
        current_record = supabase.table("documents").select("*").eq("id", uuid).execute()
        if not current_record.data:
            raise HTTPException(status_code=404, detail=f"No record found with uuid: {uuid}")

        # Plagiarism detection
        folder = current_record.data[0]["folder"]
        previous_records = supabase.table("documents").select("*").eq("folder", folder).lt("uploadedDate", uploaded_date).execute().data
        plagiarism_results = {}

        if previous_records:
            previous_embeddings = []
            for record in previous_records:
                if record.get("embedding"):
                    try:
                        if isinstance(record["embedding"], str):
                            embedding = [float(x) for x in json.loads(record["embedding"])]
                        else:
                            embedding = [float(x) for x in record["embedding"]]
                        previous_embeddings.append(embedding)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue

            if previous_embeddings:
                current_embedding = vector if isinstance(vector, list) else [float(x) for x in json.loads(vector)]
                similarities = cosine_similarity(np.array([current_embedding]), np.array(previous_embeddings))[0]
                similarity_list = [
                    (r["nameStudent"] or "Unknown", float(sim))
                    for r, sim in zip([r for r in previous_records if r.get("embedding")], similarities)
                ]
                top_2 = sorted(similarity_list, key=lambda x: x[1], reverse=True)[:2]
                plagiarism_results = dict(top_2)

        # Calculate time difference in minutes between deadline and uploaded date
        deadline = current_record.data[0]["deadline"]
        uploaded_date = current_record.data[0]["uploadedDate"]
        time_diff = (deadline - uploaded_date).total_seconds() / 3600

        # Clustering
        data_prediction = pd.DataFrame({
            'sentences': [sentence_count],
            'page': [page_count],
            'timing': [time_diff],
            'plagiarism': [plagiarism_results]
        })

        data_prediction = StandardScaler().fit_transform(data_prediction)
        clustering = loaded_model.fit_predict(data_prediction)

        # Update to database
        response = supabase.table("documents").update({
            "nameStudent": extracted_data["Name"] or "null",
            "NRP": extracted_data["ID"],
            "isiTugas": markdown_content,
            "embedding": vector,
            "page": page_count,
            "sentences": sentence_count,
            "plagiarism": plagiarism_results,
            "clustering": clustering,
            "plagiarism": plagiarism_results,
            "clustering": data_prediction
        }).eq("id", uuid).execute()

        # Error handling
        if not response.data:
            raise HTTPException(status_code=404, detail=f"No record found with uuid: {uuid}")
        
        # Variable return
        return {
            "message": "File processed and record updated successfully in Supabase.",
            "extracted_entities": extracted_data,
            "vector": vector,
            "page_count": page_count,
            "sentence_count": sentence_count,
            "plagiarism_results": plagiarism_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Model request untuk agent feedback
class AssignmentMetaModel(BaseModel):
    title: str
    description: str

class AssignmentRequest(BaseModel):
    assignment_meta: AssignmentMetaModel
    assignment_content: str

# Route baru untuk memproses feedback assignment
@app.post("/agent-feedback")
async def agent_feedback_endpoint(request: AssignmentRequest):
    try:
        feedback_result = process_assignment(
            assignment_meta=request.assignment_meta.dict(),
            assignment_content=request.assignment_content
        )
        return feedback_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))