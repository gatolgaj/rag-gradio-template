from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from qdrant_client import QdrantClient
from langchain import OpenAI

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI
openai.api_key = "your_openai_api_key"
openai_model = OpenAI(model="text-davinci-003")

# Initialize Qdrant
qdrant_client = QdrantClient(url="http://localhost:6333")
collection_name = "your_collection"

class QueryRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(request: QueryRequest):
    prompt = request.prompt
    # Retrieve relevant context from Qdrant
    try:
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=openai_model.embed(prompt),
            top=5
        )
        context = " ".join([result["payload"]["text"] for result in results])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Generate response using OpenAI
    try:
        response = openai_model.generate(prompt, context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"response": response}
