# 🎓 PokéGuru — AI Pokémon Mentor

> **Professor Oak speaks your language!**  
> An AI-powered Pokémon team mentor supporting 22 Indian languages, deep strategic reasoning (type coverage, physical vs. special balance, hazard control, and roles), and a minimal Apple-inspired **Bento AI** design.

---

## 🌟 Features

- **Multilingual Support**: Ask questions in Telugu, Hindi, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, English, and more.
- **Deep Tactical Reasoning**:
  - Compound weakness detection (e.g. 4x Rock, Ground sweeps)
  - Physical vs. Special damage bias analysis
  - Defensive bulk & hazard vulnerability checks
  - Tailored recommendations that fit your specific roster instead of generic answers
  - Beginner vs. Competitive advice modes
- **Bento AI Dark/Light Design**: Clean, minimal black-and-white grid layout with animated stat bars, interactive type badges, and retro Pokémon art.
- **PokéAPI Integration**: Real-time stats, abilities, sprites, and artwork for all 898+ Pokémon.
- **Sarvam AI Powered**:
  - **Mayura**: Translation across Indian languages
  - **Sarvam 105B**: Advanced strategic reasoning and team balance evaluation
  - **Bulbul**: Speech synthesis / audio response

---

## 🔄 Architecture Pipeline

```text
User Question (e.g. Telugu)
         │
         ▼
Sarvam Mayura Translation (Telugu ➔ English)
         │
         ▼
Intent & Entity Extraction (Extract Pokémon names)
         │
         ▼
PokéAPI (Stats, Types, Abilities, Sprites)
         │
         ▼
Strategic Team Analyzer (Weakness chart, bulk check, role coverage)
         │
         ▼
Sarvam 105B LLM Reasoning (Professor Oak tactical advice)
         │
         ▼
Sarvam Mayura Translation (English ➔ Telugu / Selected Language)
         │
         ▼
Natural Indic Explanation on Bento UI
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/suryatejauppala9-hash/pokepok.git
cd pokepok
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python run.py
```
Or directly with uvicorn:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:  
👉 **`http://localhost:8000`**

---

## ☁️ Deployment

### Option A: One-Click on Render
1. Create a new **Web Service** on [Render.com](https://render.com).
2. Connect your GitHub repository: `suryatejauppala9-hash/pokepok`.
3. Set the following settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. *(Optional)* Add Environment Variables in Render:
   - `SARVAM_API_KEY`: Your Sarvam AI key
   - `GROQ_API_KEY`: Optional Groq key for fallback

### Option B: Deploy on Railway
1. Go to [Railway.app](https://railway.app).
2. Select **Deploy from GitHub repo** and pick `pokepok`.
3. Railway automatically detects `requirements.txt` and `Procfile`.
4. Your service will be live with an instant public URL!

### Option C: Docker
```bash
docker build -t pokeguru .
docker run -p 8000:8000 pokeguru
```

---

## ⚙️ Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SARVAM_API_KEY` | Sarvam AI API subscription key | Pre-configured |
| `GROQ_API_KEY` | Optional Groq API key for alternative LLM | `None` (Optional) |
| `PORT` | Port for the web server | `8000` |

---

## 📖 API Documentation

Once running, interactive OpenAPI documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key Endpoints:
- `POST /api/query`: Submits trainer question, runs analysis pipeline, returns recommendations and native language advice.
- `GET /api/pokemon/{name}`: Fetches raw Pokémon data directly from PokéAPI.
- `POST /api/tts`: Converts text advice into speech audio using Sarvam Bulbul.
- `GET /api/health`: Health status endpoint.

---

## 🛡️ License
MIT License. Built for trainers across India and beyond!
