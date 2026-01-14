from flask import Flask, request, jsonify, render_template
from groq import Groq
import requests, io, base64
import soundfile as sf
import torch

from transformers import pipeline, VitsModel, AutoTokenizer

# ===================== INIT =====================
app = Flask(__name__, static_folder="static")
print("🔥 SERVER FILE LOADED 🔥")

# ===================== API KEYS =====================
NVIDIA_API_KEY = ""
GROQ_API_KEY = ""

# ===================== CLIENTS =====================
stt_client = Groq(api_key=GROQ_API_KEY)
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# ===================== LANGUAGE MAPS =====================
WHISPER_LANG_MAP = {
    # ISO codes
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "bn": "Bengali",

    # Full names returned by Whisper
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
def is_greeting(text):
    return text.strip().lower() in [
        "hi", "hello", "hey", "ok", "okay", "thanks", "thank you"
    ]

# ===================== STT =====================
def speech_to_text(file):
    audio_bytes = file.read()
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

def llama_request(user_text):
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

# ===================== force_text_to_language =====================
def force_text_to_language(text, language):
    return translator(
        text,
        src_lang="auto",
        tgt_lang=LANGUAGE_CODES[language],
        max_length=512
    )[0]["translation_text"]


# ===================== TRANSLATE ANSWER ONLY =====================
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

# ===================== ROUTES =====================
@app.route("/")
def index():
    return render_template("index.html")

# ===================== TEXT CHAT (DROPDOWN) =====================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_text = data.get("text", "")
    target_language = data.get("language", "English")

    if is_greeting(user_text):
        return jsonify({"text": "Hi! How can I help you?"})

    answer = llama_request(user_text)

    final_answer = translate_answer(answer, target_language)
    return jsonify({"text": final_answer})

# ===================== TEXT CHAT AUTO =====================
@app.route("/chat-auto", methods=["POST"])
def chat_auto():
    data = request.get_json()
    user_text = data.get("text", "").strip()

    if not user_text:
        return jsonify({"text": ""})

    if is_greeting(user_text):
        return jsonify({"text": user_text})

    reply = llama_request(
        f"Reply in the same language as the user.\n\n{user_text}"
    )

    return jsonify({"text": reply})

# ===================== VOICE TRANSLATE =====================
@app.route("/voice-translate", methods=["POST"])
def voice_translate():
    file = request.files.get("file")
    target_language = request.form.get("language", "English")

    spoken_text, _ = speech_to_text(file)

    translated = translator(
        spoken_text,
        src_lang="auto",
        tgt_lang=LANGUAGE_CODES[target_language],
        max_length=512
    )[0]["translation_text"]

    return jsonify({"text": translated})

# ===================== VOICE CHAT (DROPDOWN) =====================
@app.route("/voice-chat", methods=["POST"])
def voice_chat():
    file = request.files.get("file")
    target_language = request.form.get("language", "English")

    spoken_text, _ = speech_to_text(file)

    if is_greeting(spoken_text):
        return jsonify({"text": "Hi! How can I help you?"})

    # 🔥 NO TRANSLATION BEFORE LLaMA
    answer = llama_request(spoken_text)

    final_answer = translate_answer(answer, target_language)
    return jsonify({"text": final_answer})

# ===================== VOICE CHAT AUTO =====================
@app.route("/voice-chat-auto", methods=["POST"])
def voice_chat_auto():
    file = request.files.get("file")
    print(f"file: {file}")
    print("------------------------------------------------------------")

    spoken_text, whisper_lang = speech_to_text(file)
    print(f"Spoken Text: {spoken_text}")
    print(f"Whisper Language: {whisper_lang}")  
    print("------------------------------------------------------------")

    language = WHISPER_LANG_MAP.get(whisper_lang.lower(), "English")
    print(f"Detected Language: {language}")
    print("------------------------------------------------------------")

    if is_greeting(spoken_text):
        return jsonify({
            "text": "Hi! How can I help you?",
            "language": language
        })

    # 1️⃣ Let LLaMA answer freely
    reply_raw = llama_request(spoken_text)
    print(f"LLaMA Raw Reply: {reply_raw}")
    print("------------------------------------------------------------")

    # 2️⃣ FORCE into Whisper language (CRITICAL)
    reply = force_text_to_language(reply_raw, language)
    print(f"Final Reply: {reply}")  
    print("------------------------------------------------------------")

    return jsonify({
        "text": reply,
        "language": language
    })

# ===================== TTS =====================
@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    text = data.get("text", "")
    language = data.get("language", "English")

    audio_buffer = mms_tts(text, language)

    # SAVE AUDIO TO FILE
    with open("output.wav", "wb") as f:
        f.write(audio_buffer.getvalue())

    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()

    return jsonify({
        "audio": "data:audio/wav;base64," + audio_base64
    })

# ===================== RUN =====================
if __name__ == "__main__":
    app.run(port=5001, debug=True, use_reloader=False)

# feature: cleaned API key handling (again changed)
# This is to check the changes that made
# This is to check the changes again made by the mahendra-branch to check the -
# - changes made will also be present in the main or not
