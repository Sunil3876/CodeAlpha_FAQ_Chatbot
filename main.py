import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI(title="FAQ ChatBot", version="2.0")

# Mounting static files for Frontend
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 1. Load Dataset (Requires faq_data.json in the same folder)
with open("faq_data.json", "r") as file:
    faq_data = json.load(file)

# Prepare documents for Vector Store
texts = [f"Question: {item['question']} Answer: {item['answer']}" for item in faq_data]
metadatas = [{"answer": item["answer"], "question": item["question"]} for item in faq_data]

# 2. Initialize AI Embeddings Model (Local & Free)
print("Loading HuggingFace Embeddings Model... Please wait...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Create High-Performance FAISS Vector Database
print("Building Vector Indexes...")
vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
print("Vector Store Ready!")

# Pydantic schema for input validation
class ChatQuery(BaseModel):
    message: str

@app.get("/")
async def read_index():
    # Modified to prevent crashes if index.html is missing
    file_path = "static/index.html"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse(
        content="<h1>Frontend Missing</h1><p>Please create an 'index.html' file inside the 'static' folder to view the interface.</p>", 
        status_code=404
    )

@app.post("/api/chat")
async def chat_endpoint(query: ChatQuery):
    try:
        user_input = query.message.strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Semantic Vector Search with Score
        results = vector_store.similarity_search_with_score(user_input, k=1)
        
        if not results:
            return {"response": "I'm sorry, I couldn't find any relevant information."}
        
        doc, score = results[0]
        
        # FAISS score standard threshold validation
        if score > 1.2:
            return {
                "response": "I am not completely sure about that. Could you please rephrase your question or contact our human support team directly?",
                "match_score": float(score),
                "confident": False
            }
        
        return {
            "response": doc.metadata["answer"],
            "matched_question": doc.metadata["question"],
            "match_score": float(score),
            "confident": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)