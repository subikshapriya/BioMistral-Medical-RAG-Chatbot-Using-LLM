from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# -----------------------------
# APP
# -----------------------------
app = FastAPI(title="Medical RAG Chatbot")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# -----------------------------
# CONFIG
# -----------------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "medical_docs"

# -----------------------------
# QDRANT
# -----------------------------
qdrant_client = QdrantClient(url=QDRANT_URL)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# LLM
# -----------------------------
llm = Ollama(
    model="tinyllama",   # change to biomistral later
    temperature=0.2
)

# -----------------------------
# PROMPT
# -----------------------------
prompt = PromptTemplate.from_template(
    """
You are a medical assistant.
Answer ONLY using the context.
If the answer is not found, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)

# -----------------------------
# RAG CHAIN
# -----------------------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# -----------------------------
# REQUEST MODEL
# -----------------------------
class QueryRequest(BaseModel):
    question: str

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
def ask(req: QueryRequest):
    try:
        result = rag_chain.invoke(req.question)

        answer = (
            result.content
            if hasattr(result, "content")
            else str(result)
        )

        return JSONResponse(content={"answer": answer})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"answer": f"❌ Backend error: {str(e)}"}
        )