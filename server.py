from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from groq import Groq
import requests, io, base64
import soundfile as sf
import torch

from transformers import pipeline, VitsModel, AutoTokenizer
import uvicorn
import webbrowser

# ===================== INIT =====================
app = FastAPI()
print("🔥 SERVER FILE LOADED (FASTAPI) 🔥")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ===================== API KEYS =====================
NVIDIA_API_KEY = " "
GROQ_API_KEY = " "

# ===================== CLIENTS =====================
stt_client = Groq(api_key=GROQ_API_KEY)
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# ===================== LANGUAGE MAPS =====================
WHISPER_LANG_MAP = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "bn": "Bengali",
    "english": "English",
    "hindi": "Hindi",
    "kannada": "Kannada",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "malayalam": "Malayalam",
    "bengali": "Bengali",
}

# ===================== LOAD NLLB =====================
print("🌐 Loading NLLB...")
translator = pipeline("translation", model="facebook/nllb-200-distilled-600M")

LANGUAGE_CODES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Malayalam": "mal_Mlym",
    "Bengali": "ben_Beng",
}

# ===================== INTENT =====================
def is_greeting(text: str) -> bool:
    return text.strip().lower() in [
        "hi", "hello", "hey", "ok", "okay", "thanks", "thank you"
    ]

# ===================== STT =====================
async def speech_to_text(file: UploadFile):
    audio_bytes = await file.read()
    res = stt_client.audio.transcriptions.create(
        file=(file.filename, audio_bytes),
        model="whisper-large-v3",
        response_format="verbose_json",
        temperature=0
    )
    lang = (res.language or "").strip().lower()
    return res.text.strip(), lang

# ===================== LLaMA =====================
SYSTEM_PROMPT = """
You are a strict assistant.
Answer ONLY what the user explicitly asks.
Do NOT introduce automobiles, cars, vehicles, automation, or auto-related topics
unless the user explicitly mentions them.
You understand all Indian languages.
"""

def llama_request(user_text: str) -> str:
    payload = {
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        "max_tokens": 256
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    res = requests.post(NVIDIA_URL, headers=headers, json=payload, timeout=60)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

# ===================== TRANSLATION =====================
def force_text_to_language(text, language):
    return translator(
        text,
        src_lang="auto",
        tgt_lang=LANGUAGE_CODES[language],
        max_length=512
    )[0]["translation_text"]

def translate_answer(text, target_language):
    return translator(
        text,
        src_lang="auto",
        tgt_lang=LANGUAGE_CODES[target_language],
        max_length=512
    )[0]["translation_text"]

# ===================== MMS TTS =====================
MMS_TTS_MODELS = {
    "English": "facebook/mms-tts-eng",
    "Hindi": "facebook/mms-tts-hin",
    "Kannada": "facebook/mms-tts-kan",
    "Tamil": "facebook/mms-tts-tam",
    "Telugu": "facebook/mms-tts-tel",
    "Malayalam": "facebook/mms-tts-mal",
    "Bengali": "facebook/mms-tts-ben",
}

TTS_CACHE = {}

def load_tts(language):
    model_id = MMS_TTS_MODELS.get(language, MMS_TTS_MODELS["English"])
    if model_id not in TTS_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(model_id)
        TTS_CACHE[model_id] = (tokenizer, model)
    return TTS_CACHE[model_id]

def mms_tts(text, language):
    tokenizer, model = load_tts(language)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        audio = model(**inputs).waveform

    buffer = io.BytesIO()
    sf.write(buffer, audio.squeeze().numpy(), 16000, format="WAV")
    buffer.seek(0)
    return buffer

# ===================== MODELS =====================
class ChatRequest(BaseModel):
    text: str
    language: str | None = "English"

class TTSRequest(BaseModel):
    text: str
    language: str = "English"

# ===================== ROUTES =====================

# ✅ EXACT FLASK BEHAVIOR
@app.get("/", response_class=HTMLResponse)
def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

# ===================== TEXT CHAT =====================
@app.post("/chat")
def chat(req: ChatRequest):
    if is_greeting(req.text):
        return {"text": "Hi! How can I help you?"}

    answer = llama_request(req.text)
    final_answer = translate_answer(answer, req.language)
    return {"text": final_answer}

# ===================== TEXT CHAT AUTO =====================
@app.post("/chat-auto")
def chat_auto(req: ChatRequest):
    if not req.text.strip():
        return {"text": ""}

    if is_greeting(req.text):
        return {"text": req.text}

    reply = llama_request(
        f"Reply in the same language as the user.\n\n{req.text}"
    )
    return {"text": reply}

# ===================== VOICE TRANSLATE =====================
@app.post("/voice-translate")
async def voice_translate(
    file: UploadFile = File(...),
    language: str = Form("English")
):
    spoken_text, _ = await speech_to_text(file)
    translated = translate_answer(spoken_text, language)
    return {"text": translated}

# ===================== VOICE CHAT =====================
@app.post("/voice-chat")
async def voice_chat(
    file: UploadFile = File(...),
    language: str = Form("English")
):
    spoken_text, _ = await speech_to_text(file)

    if is_greeting(spoken_text):
        return {"text": "Hi! How can I help you?"}

    answer = llama_request(spoken_text)
    final_answer = translate_answer(answer, language)
    return {"text": final_answer}

# ===================== VOICE CHAT AUTO =====================
@app.post("/voice-chat-auto")
async def voice_chat_auto(file: UploadFile = File(...)):
    spoken_text, whisper_lang = await speech_to_text(file)

    language = WHISPER_LANG_MAP.get(whisper_lang.lower(), "English")

    if is_greeting(spoken_text):
        return {"text": "Hi! How can I help you?", "language": language}

    reply_raw = llama_request(spoken_text)
    reply = force_text_to_language(reply_raw, language)

    return {"text": reply, "language": language}

# ===================== TTS =====================
@app.post("/tts")
def tts(req: TTSRequest):
    audio_buffer = mms_tts(req.text, req.language)

    with open("output.wav", "wb") as f:
        f.write(audio_buffer.getvalue())

    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()

    return {"audio": "data:audio/wav;base64," + audio_base64}

# ===================== RUN =====================
if __name__ == "__main__":
    print("🚀 Running on http://localhost:5001")
    webbrowser.open("http://localhost:5001")
    uvicorn.run(app, host="0.0.0.0", port=5001, reload=False)


# feature: cleaned API key handling (again changed)
# This is the changes made by the mahendra-branch code