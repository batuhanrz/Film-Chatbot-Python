import os
import csv
import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="CineBot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --- CSV'den ortalama rating hesapla ---
def load_avg_ratings(csv_path="film.csv"):
    ratings = defaultdict(list)
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('movie_name', '').strip()
                try:
                    ratings[name].append(float(row.get('Ratings', 0)))
                except:
                    pass
        return {name: round(sum(vals)/len(vals), 1) for name, vals in ratings.items()}
    except Exception as e:
        print(f"CSV okunamadi: {e}")
        return {}

# --- RAG ---
class MovieRAG:
    def __init__(self, api_key):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)
        self.vectorstore = None
        self.retriever = None
        self.avg_ratings = {}

    def load(self, faiss_path="langchain_faiss_db", csv_path="film.csv"):
        self.vectorstore = LangChainFAISS.load_local(
            faiss_path, self.embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.5}
        )
        self.avg_ratings = load_avg_ratings(csv_path)
        print(f"RAG yuklendi. {len(self.avg_ratings)} film icin ortalama rating hazir.")

    def query(self, question: str, history: list):
        history_text = ""
        for msg in history[-6:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history_text += f"Kullanici: {content}\n"
            elif role == "assistant":
                history_text += f"Asistan: {content}\n"

        # Sohbet geçmişinden bahsedilen filmleri bul
        mentioned_films = []
        all_messages = history + [{"role": "user", "content": question}]
        for msg in all_messages:
            content_text = msg.get("content", "")
            # FAISS'te bu filmleri ara
            try:
                film_docs = self.vectorstore.similarity_search(content_text, k=3)
                for fd in film_docs:
                    title = fd.metadata.get("title", "")
                    genre = fd.metadata.get("genre", "")
                    emotion = fd.metadata.get("emotion", "")
                    if title and genre:
                        mentioned_films.append(f"{genre} {emotion}")
            except:
                pass

        # Zenginleştirilmiş sorgu: orijinal soru + bulunan film özellikleri
        enriched_query = question
        if mentioned_films:
            enriched_query = question + " " + " ".join(mentioned_films[:3])

        docs = self.retriever.invoke(enriched_query)

        context = "\n\n".join([
            f"Film: {d.metadata.get('title','')}\nTur: {d.metadata.get('genre','')}\nDuygu: {d.metadata.get('emotion','')}\n{d.page_content[:200]}"
            for d in docs
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen sicak, samimi bir film oneri asistanisin.

GOREV:
- Kullanici belirli filmlerden bahsediyorsa, o filmlerin turunu, duygusunu ve temalarini analiz et.
- Sonra bu ozelliklere benzer filmler oner.
- Kullanici genel bir tur istiyorsa o tura gore oner.

FORMAT:
1. 1-2 cumlelik giris: kullanicinin bahsettigi filmlerin ortak ozelliklerini belirt (tur, tema, duygu).
2. "Bunlara benzer filmler oneririm:" de.
3. Her film icin:
**Film Adi**
Bu filmi neden onerdiklerini 1 cumleyle acikla (onceki filmlerle baglantisi varsa belirt).
4. Kapanista baska tur veya tema teklif et.
5. Turkce, samimi ve dogal yaz.

Sohbet gecmisi:
{history}

Bulunan filmler (bunlardan sec):
{context}"""),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"question": question, "context": context, "history": history_text})

        seen = set()
        unique_docs = []
        for d in docs:
            title = d.metadata.get("title", "")
            if title not in seen:
                seen.add(title)
                # CSV'den ortalama rating al
                avg_rating = self.avg_ratings.get(title, None)
                unique_docs.append({
                    "title": title,
                    "genre": str(d.metadata.get("genre", "")).replace("[","").replace("]","").replace("'",""),
                    "rating": avg_rating,
                    "emotion": d.metadata.get("emotion", "")
                })

        return {"answer": answer, "movies": unique_docs[:5]}


api_key = os.getenv("OPENAI_API_KEY")
rag = MovieRAG(api_key)

@app.on_event("startup")
async def startup():
    rag.load("langchain_faiss_db", "film.csv")

class ChatRequest(BaseModel):
    question: str
    history: List[dict] = []

@app.post("/api/chat")
async def chat(req: ChatRequest):
    return rag.query(req.question, req.history)

@app.get("/api/health")
async def health():
    return {"status": "ok", "loaded": rag.vectorstore is not None, "ratings": len(rag.avg_ratings)}

@app.get("/")
async def index():
    return FileResponse("index.html")