let mediaRecorder = null;
let recordedChunks = [];

document.addEventListener("DOMContentLoaded", () => {

  // ===================== ELEMENTS =====================
  const startBtn = document.getElementById("start");
  const stopBtn = document.getElementById("stop");
  const output = document.getElementById("output");
  const fileInput = document.getElementById("audioFile");
  const uploadBtn = document.getElementById("uploadBtn");
  const languageSelect = document.getElementById("language");

  const chatInput = document.getElementById("chatInput");
  const chatOutput = document.getElementById("chatOutput");
  const sendChatBtn = document.getElementById("sendChat");

  const autoLangCheckbox = document.getElementById("autoLang");
  const voiceModeRadios = document.getElementsByName("voiceMode");

  // ===================== HELPERS =====================
  function getVoiceMode() {
    for (const r of voiceModeRadios) {
      if (r.checked) return r.value;
    }
    return "translate";
  }

  function resetUI(msg) {
    output.innerText = msg;
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }

  // ===================== START RECORDING =====================
  startBtn.onclick = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunks = [];

      mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm"
      });

      mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) recordedChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(recordedChunks, { type: "audio/webm" });
        sendVoice(audioBlob);
      };

      mediaRecorder.start();

      startBtn.disabled = true;
      stopBtn.disabled = false;
      output.innerText = "🎙️ Recording...";

    } catch (err) {
      console.error("Mic error:", err);
      resetUI("❌ Microphone access denied");
    }
  };

  // ===================== STOP RECORDING =====================
  stopBtn.onclick = () => {
    if (!mediaRecorder) return;

    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());

    mediaRecorder = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
  };

  // ===================== UPLOAD AUDIO =====================
  uploadBtn.onclick = () => {
    const file = fileInput.files[0];
    if (!file) {
      output.innerText = "❌ Please select an audio file";
      return;
    }
    sendVoice(file);
  };

  // ===================== SEND VOICE =====================
  function sendVoice(blob) {
    const form = new FormData();
    form.append("file", blob, "recording.webm");
    form.append("language", languageSelect.value);

    const mode = getVoiceMode();
    const endpoint =
      mode === "translate" ? "/voice-translate" :
      mode === "chat-dropdown" ? "/voice-chat" :
      "/voice-chat-auto";

    output.innerText = "⏳ Processing voice...";

    fetch(endpoint, { method: "POST", body: form })
      .then(res => res.json())
      .then(data => {
        if (!data.text) {
          output.innerText = "❌ No response";
          return;
        }

        output.innerText = data.text;

        const ttsLanguage =
          mode === "chat-auto"
            ? (data.language || "English")
            : languageSelect.value;

        playTTS(data.text, ttsLanguage);
      })
      .catch(err => {
        console.error("Voice error:", err);
        resetUI("❌ Server error");
      });
  }

  // ===================== PLAY TTS =====================
  function playTTS(text, language) {
    if (!text.trim()) return;

    fetch("/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, language })
    })
      .then(res => res.json())
      .then(data => {
        if (!data.audio) {
          console.error("TTS returned empty audio");
          return;
        }

        const audio = new Audio(data.audio);
        audio.play().catch(() => {
          document.body.addEventListener(
            "click",
            () => audio.play().catch(() => {}),
            { once: true }
          );
        });
      })
      .catch(err => console.error("TTS error:", err));
  }

  // ===================== TEXT CHAT =====================
  function sendChatMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    const autoMode = autoLangCheckbox.checked;
    const endpoint = autoMode ? "/chat-auto" : "/chat";
    const payload = autoMode
      ? { text }
      : { text, language: languageSelect.value };

    chatOutput.innerText = "⏳ Thinking...";

    fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(res => res.json())
      .then(data => {
        chatOutput.innerText = data.text || "❌ No response";
      })
      .catch(() => {
        chatOutput.innerText = "❌ Server error";
      });

    chatInput.value = "";
  }

  sendChatBtn.addEventListener("click", sendChatMessage);

  chatInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });
});

