# 🎬 CineBot — AI Destekli Film Öneri Chatbotu

CineBot, Retrieval-Augmented Generation (RAG) mimarisi kullanarak kullanıcıların doğal dilde ifade ettiği tercihlerine göre kişiselleştirilmiş film önerileri sunan yapay zeka destekli bir chatbot uygulamasıdır.

---

## 🚀 Özellikler

- 🔍 **Semantik Arama** — FAISS vektör veritabanı ile 900.000+ film içinde anlamlı arama
- 🤖 **RAG Mimarisi** — LangChain + OpenAI GPT-4o-mini ile bağlama duyarlı yanıtlar
- 💬 **Konuşma Geçmişi** — Önceki mesajları hatırlayan çok turlu diyalog desteği
- 🎭 **Duygu & Tür Filtreleme** — Film türü, duygu tonu ve temaya göre öneri
- ⭐ **IMDb Rating** — CSV'den hesaplanan ortalama kullanıcı puanları
- 🖥️ **Modern Arayüz** — Özel tasarımlı sinema temalı web UI (FastAPI + HTML)

---

## 📁 Proje Yapısı

```
Film-Chatbot-Python/
│
├── main.py                    # FastAPI backend — RAG sorgu motoru
├── index.html                 # Frontend — Sinema temalı chat arayüzü
├── rebuild_faiss.py           # FAISS vektör veritabanını yeniden oluşturur
├── filmkod.ipynb              # Geliştirme notebook'u
├── film.csv                   # Ham veri seti (900K+ film kaydı)
├── langchain_faiss_db/        # FAISS vektör veritabanı
│   ├── index.faiss
│   └── index.pkl
├── movie_embeddings_full.pkl  # Önceden hesaplanmış embedding'ler
├── requirements.txt           # Python bağımlılıkları
└── sunum/                     # Proje sunum dosyaları
```

---

## 🛠️ Kullanılan Teknolojiler

| Katman | Teknoloji |
|---|---|
| Backend | FastAPI, Python 3.11 |
| LLM | OpenAI GPT-4o-mini |
| Embedding | OpenAI text-embedding-3-small (1536 dim) |
| Vektör DB | FAISS (IndexFlatL2) |
| RAG Framework | LangChain (MMR retrieval) |
| Frontend | Vanilla HTML/CSS/JS |

---

## ⚙️ Kurulum

### 1. Repoyu klonla

```bash
git clone https://github.com/batuhanrz/Film-Chatbot-Python.git
cd Film-Chatbot-Python
```

### 2. Sanal ortam oluştur ve bağımlılıkları yükle

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. `.env` dosyasını oluştur

```
OPENAI_API_KEY=sk-...
```

### 4. Uygulamayı başlat

```bash
uvicorn main:app --reload
```

Tarayıcıda `http://localhost:8000` adresini aç.

---

## 🧠 Nasıl Çalışır?

1. Kullanıcı doğal dilde bir film tercihi yazar ("Hüzünlü ama güzel bir film istiyorum")
2. Soru, OpenAI embedding modeli ile vektöre dönüştürülür
3. FAISS, MMR algoritmasıyla en alakalı ve çeşitli 10 filmi bulur
4. GPT-4o-mini, bulunan filmler ve konuşma geçmişini kullanarak Türkçe öneri üretir
5. Film kartları, tür, duygu ve IMDb puanı ile birlikte gösterilir

---

## 📊 Veri Seti

- **Kaynak:** IMDb film veritabanı
- **Boyut:** 900.000+ kayıt
- **Özellikler:** `movie_name`, `genres`, `Reviews`, `Ratings`, `emotion`, `Description`
- **Embedding süresi:** ~2-3 saat (OpenAI API ile)

---

## 🗂️ Retrieval Stratejisi

**MMR (Maximum Marginal Relevance)** algoritması iki kriteri dengeler:
- **Relevance** — Sorguyla semantik benzerlik
- **Diversity** — Önerilen filmler arasında çeşitlilik

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.5}
)
```

---

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.
