/**
 * main.js — PlantGuard AI
 * Futuristic Cyber-Botanical Interactive Client Controller
 * 
 * Features:
 *   1. Drag-and-drop & Instant Image Preview
 *   2. Live Webcam / Mobile Camera Viewfinder with Leaf Reticle & Audio Shutter
 *   3. Multi-View Bio-Optical Lesion Inspector
 *   4. Text-to-Speech Voice Narration (Web Speech API)
 *   5. Printable Clinical Prescription Engine
 *   6. Virtual AI Agronomist Interactive Consultation
 */

(function () {
  "use strict";

  // ── 1. Synthetic Audio Engine (Web Audio API) ───────────────────────────────
  const audioCtx = (window.AudioContext || window.webkitAudioContext) ? new (window.AudioContext || window.webkitAudioContext)() : null;

  function playShutterSound() {
    if (!audioCtx) return;
    try {
      if (audioCtx.state === "suspended") audioCtx.resume();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(800, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(120, audioCtx.currentTime + 0.08);
      gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
      gain.gain.linearRampToValueAtTime(0.01, audioCtx.currentTime + 0.08);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.09);
    } catch (e) {
      // Audio autoplay policy fallback
    }
  }

  // ── 2. Mode Navigation Tabs ────────────────────────────────────────────────
  const modeTabs = document.querySelectorAll(".mode-tab");
  modeTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      modeTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetPanelId = tab.getAttribute("data-panel");
      document.querySelectorAll(".tab-panel").forEach(panel => {
        panel.classList.remove("active");
      });
      const targetPanel = document.getElementById(targetPanelId);
      if (targetPanel) targetPanel.classList.add("active");

      // Stop camera if user switches away from webcam panel
      if (targetPanelId !== "panel-camera" && window.plantCameraStream) {
        stopWebcam();
      }
    });
  });

  // ── 3. File Upload & Drag-and-Drop ──────────────────────────────────────────
  const dropZone     = document.getElementById("drop-zone");
  const fileInput    = document.getElementById("file-input");
  const previewArea  = document.getElementById("preview-area");
  const previewImg   = document.getElementById("preview-img");
  const fileNameEl   = document.getElementById("file-name");
  const fileSizeEl   = document.getElementById("file-size");
  const clearBtn     = document.getElementById("clear-btn");
  const uploadForm   = document.getElementById("upload-form");
  const submitBtn    = document.getElementById("submit-btn");
  const progressWrap = document.getElementById("progress-wrap");
  const progressLabel= document.getElementById("progress-label");

  if (dropZone && fileInput) {
    ["dragenter", "dragover"].forEach(evt => {
      dropZone.addEventListener(evt, e => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
      });
    });

    ["dragleave", "drop"].forEach(evt => {
      dropZone.addEventListener(evt, e => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
      });
    });

    dropZone.addEventListener("drop", e => {
      const files = e.dataTransfer?.files;
      if (files && files.length) {
        fileInput.files = files;
        handleFileSelection(files[0]);
      }
    });

    dropZone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) {
        handleFileSelection(fileInput.files[0]);
      }
    });

    if (clearBtn) {
      clearBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.value = "";
        previewImg.src = "";
        previewArea.classList.add("hidden");
        dropZone.classList.remove("hidden");
      });
    }

    if (uploadForm) {
      uploadForm.addEventListener("submit", () => {
        if (!fileInput.files.length) return;
        if (submitBtn) {
          submitBtn.disabled = true;
          submitBtn.textContent = "⚙️ Scanning Botanical Signatures...";
        }
        if (progressWrap) {
          progressWrap.classList.remove("hidden");
          cycleProgressMessages();
        }
      });
    }
  }

  function handleFileSelection(file) {
    if (!file) return;
    const allowed = ["image/jpeg", "image/png", "image/bmp", "image/webp"];
    if (!allowed.includes(file.type)) {
      showToastNotification("Please select a valid image file (JPG, PNG, or WEBP).", "error");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      showToastNotification("File exceeds the 15 MB limit.", "error");
      return;
    }

    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      if (fileNameEl) fileNameEl.textContent = file.name;
      if (fileSizeEl) fileSizeEl.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";
      if (previewArea) previewArea.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
  }

  function cycleProgressMessages() {
    const messages = [
      "Verifying botanical leaf presence & chlorophyll signatures...",
      "Analyzing cellular tissue & segmenting pathogen lesions...",
      "Extracting 114 color, GLCM texture, and contour descriptors...",
      "Evaluating with Random Forest, SVM & KNN classifiers...",
      "Formulating agronomic prescription & fertilizer ratios..."
    ];
    let i = 0;
    setInterval(() => {
      i = (i + 1) % messages.length;
      if (progressLabel) progressLabel.textContent = messages[i];
    }, 1200);
  }

  // ── 4. Live Webcam Module ──────────────────────────────────────────────────
  const startCameraBtn = document.getElementById("start-camera-btn");
  const stopCameraBtn  = document.getElementById("stop-camera-btn");
  const flipCameraBtn  = document.getElementById("flip-camera-btn");
  const snapBtn        = document.getElementById("snap-btn");
  const videoEl        = document.getElementById("webcam-video");
  const canvasEl       = document.getElementById("webcam-canvas");
  const placeholderEl  = document.getElementById("camera-placeholder");
  const controlsEl     = document.getElementById("camera-controls");
  const cameraFlash    = document.getElementById("camera-flash");
  const cameraDataInput= document.getElementById("camera-data-input");
  const cameraForm     = document.getElementById("camera-form");

  let currentFacingMode = "environment";

  if (startCameraBtn) {
    startCameraBtn.addEventListener("click", () => startWebcam());
  }

  if (stopCameraBtn) {
    stopCameraBtn.addEventListener("click", () => stopWebcam());
  }

  if (flipCameraBtn) {
    flipCameraBtn.addEventListener("click", () => {
      currentFacingMode = currentFacingMode === "environment" ? "user" : "environment";
      stopWebcam();
      startWebcam();
    });
  }

  if (snapBtn) {
    snapBtn.addEventListener("click", () => captureWebcamSnap());
  }

  async function startWebcam() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToastNotification("Camera API is not supported in this browser.", "error");
      return;
    }

    try {
      const constraints = {
        video: {
          facingMode: currentFacingMode,
          width: { ideal: 1280 },
          height: { ideal: 960 }
        }
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      window.plantCameraStream = stream;
      if (videoEl) {
        videoEl.srcObject = stream;
        videoEl.play();
      }
      if (placeholderEl) placeholderEl.classList.add("hidden");
      if (controlsEl) controlsEl.classList.remove("hidden");
    } catch (err) {
      showToastNotification("Camera access denied or camera device not found.", "error");
    }
  }

  function stopWebcam() {
    if (window.plantCameraStream) {
      window.plantCameraStream.getTracks().forEach(track => track.stop());
      window.plantCameraStream = null;
    }
    if (videoEl) videoEl.srcObject = null;
    if (placeholderEl) placeholderEl.classList.remove("hidden");
    if (controlsEl) controlsEl.classList.add("hidden");
  }

  function captureWebcamSnap() {
    if (!videoEl || !canvasEl || !cameraForm) return;

    // Trigger synthetic audio shutter & flash effect
    playShutterSound();
    if (cameraFlash) {
      cameraFlash.classList.add("flash-active");
      setTimeout(() => cameraFlash.classList.remove("flash-active"), 350);
    }

    canvasEl.width = videoEl.videoWidth || 640;
    canvasEl.height = videoEl.videoHeight || 480;
    const ctx = canvasEl.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

    const dataUrl = canvasEl.toDataURL("image/jpeg", 0.92);
    if (cameraDataInput) cameraDataInput.value = dataUrl;

    if (snapBtn) {
      snapBtn.disabled = true;
      snapBtn.textContent = "⏳ Analysing Captured Leaf...";
    }

    // Submit form after flash
    setTimeout(() => {
      cameraForm.submit();
    }, 200);
  }

  // ── 5. Multi-View Bio-Optical Inspector (Result Page) ──────────────────────
  const viewTabs = document.querySelectorAll(".view-tab");
  const hudTag   = document.getElementById("hud-mode-name");
  const captionEl= document.getElementById("inspector-caption");

  if (viewTabs.length) {
    const viewDescriptions = {
      orig: "Displaying high-definition original leaf photograph. Toggle tabs above to view highlighted infection lesions or vegetation density.",
      lesions: "🔴 Active Pathogen Heatmap: Computer vision isolates necrotic and chlorotic lesion clusters in bright amber-crimson contour boundaries.",
      exg: "🟢 Excess Green (ExG) Bio-Map: Normalized false-color chlorophyll spectrum highlighting photosynthetically active healthy leaf tissue."
    };

    viewTabs.forEach(tab => {
      tab.addEventListener("click", () => {
        viewTabs.forEach(t => t.classList.remove("active"));
        tab.classList.add("active");

        const target = tab.getAttribute("data-target");
        document.querySelectorAll(".inspector-img").forEach(img => img.classList.remove("active"));

        const activeImg = document.getElementById(`view-img-${target}`);
        if (activeImg) activeImg.classList.add("active");

        if (hudTag) {
          hudTag.textContent = target === "orig" ? "Original Photograph" : (target === "lesions" ? "Lesion Pathogen Heatmap" : "Chlorophyll Bio-Map (ExG)");
        }
        if (captionEl && viewDescriptions[target]) {
          captionEl.textContent = viewDescriptions[target];
        }
      });
    });
  }

  // ── 6. Audio Voice Readout (Web Speech API) ────────────────────────────────
  const voiceBtn  = document.getElementById("voice-btn");
  const voiceIcon = document.getElementById("voice-icon");
  const voiceText = document.getElementById("voice-text");
  let isSpeaking = false;

  if (voiceBtn && "speechSynthesis" in window) {
    const script = document.body.getAttribute("data-speech-text") || "";

    voiceBtn.addEventListener("click", () => {
      if (isSpeaking) {
        window.speechSynthesis.cancel();
        isSpeaking = false;
        voiceIcon.textContent = "🔊";
        voiceText.textContent = "Listen to Diagnosis";
      } else {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(script);
        utter.rate = 0.95;
        utter.pitch = 1.0;

        utter.onstart = () => {
          isSpeaking = true;
          voiceIcon.textContent = "⏸️";
          voiceText.textContent = "Pause Audio";
        };
        utter.onend = utter.onerror = () => {
          isSpeaking = false;
          voiceIcon.textContent = "🔊";
          voiceText.textContent = "Listen to Diagnosis";
        };

        window.speechSynthesis.speak(utter);
      }
    });
  }

  // ── 7. Printable Medical Rx Certificate ───────────────────────────────────
  const printRxBtn = document.getElementById("print-rx-btn");
  if (printRxBtn) {
    printRxBtn.addEventListener("click", () => {
      window.print();
    });
  }

  // ── 8. AI Agronomist Interactive Consultation ─────────────────────────────
  const chatForm   = document.getElementById("chat-form");
  const chatInput  = document.getElementById("chat-input");
  const chatThread = document.getElementById("chat-thread");
  const faqChips   = document.querySelectorAll(".faq-chip");

  if (chatForm && chatInput && chatThread) {
    chatForm.addEventListener("submit", e => {
      e.preventDefault();
      const text = chatInput.value.trim();
      if (!text) return;
      sendMessageToAgronomist(text);
      chatInput.value = "";
    });

    faqChips.forEach(chip => {
      chip.addEventListener("click", () => {
        const query = chip.getAttribute("data-q");
        if (query) sendMessageToAgronomist(query);
      });
    });

    async function sendMessageToAgronomist(message) {
      appendChatMessage(message, "user");

      // Typing placeholder
      const typingEl = appendChatMessage("Consulting Plant Doctor...", "assistant typing");

      try {
        const resp = await fetch("/api/chat_agronomist", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: message })
        });
        const data = await resp.json();
        typingEl.remove();
        appendChatMessage(data.reply || "Could not retrieve recommendation.", "assistant");
      } catch (err) {
        typingEl.remove();
        appendChatMessage("Network error connecting to Agronomist service.", "assistant");
      }
    }

    function appendChatMessage(text, sender) {
      const msg = document.createElement("div");
      msg.className = `chat-msg chat-msg--${sender.includes("user") ? "user" : "assistant"}`;
      const bubble = document.createElement("div");
      bubble.className = "chat-bubble";
      bubble.innerHTML = text.replace(/\n/g, "<br/>").replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
      msg.appendChild(bubble);
      chatThread.appendChild(msg);
      chatThread.scrollTop = chatThread.scrollHeight;
      return msg;
    }
  }

  // ── Toast Notification Helper ──────────────────────────────────────────────
  function showToastNotification(msg, type = "info") {
    const toast = document.createElement("div");
    toast.className = `flash flash--${type}`;
    toast.style.cssText = "position:fixed;top:85px;right:24px;z-index:99999;max-width:380px;box-shadow:0 12px 30px rgba(0,0,0,0.5);";
    toast.innerHTML = `<span>${type === "error" ? "⚠️" : "ℹ️"}</span> ${msg}`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4500);
  }

})();
