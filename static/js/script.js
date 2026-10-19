const chatArea = document.getElementById("chatArea");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("sendButton");
const heroSection = document.getElementById("heroSection");
const suggestionSection = document.getElementById("suggestionSection");
const historyList = document.getElementById("historyList");
const homeButton = document.getElementById("homeButton");
const historyButton = document.getElementById("historyButton");

let currentConversationId = null;

// ===== Render pesan ke #chatArea =====
function renderMessage(role, content, time) {
  const row = document.createElement("div");
  row.className = `message-row ${role === "user" ? "user" : "bot"}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "🧑" : "🤖";

  const bubble = document.createElement("div");
  bubble.className = "message";

  const contentEl = document.createElement("div");
  contentEl.className = "message-content";
  // marked.js untuk render markdown dari jawaban bot (list, bold, dll)
  contentEl.innerHTML =
    typeof marked !== "undefined" ? marked.parse(content) : content;

  const metaEl = document.createElement("div");
  metaEl.className = "meta";
  metaEl.textContent = time || "";

  bubble.appendChild(contentEl);
  bubble.appendChild(metaEl);

  if (role === "user") {
    row.appendChild(bubble);
    row.appendChild(avatar);
  } else {
    row.appendChild(avatar);
    row.appendChild(bubble);
  }

  chatArea.appendChild(row);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function showTypingIndicator() {
  const row = document.createElement("div");
  row.className = "message-row bot";
  row.id = "typingIndicator";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "🤖";

  const bubble = document.createElement("div");
  bubble.className = "message";
  bubble.innerHTML = `<div class="typing"><span></span><span></span><span></span></div>`;

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatArea.appendChild(row);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

function hideHeroIfNeeded() {
  if (heroSection) heroSection.classList.add("hidden");
  if (suggestionSection) suggestionSection.classList.add("hidden");
}

function showHero() {
  if (heroSection) heroSection.classList.remove("hidden");
  if (suggestionSection) suggestionSection.classList.remove("hidden");
}

// ===== Kirim pesan =====
async function sendMessage(text) {
  const message = (text || questionInput.value).trim();
  if (!message) return;

  hideHeroIfNeeded();
  renderMessage("user", message, currentTime());
  questionInput.value = "";
  sendButton.disabled = true;
  showTypingIndicator();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        conversation_id: currentConversationId,
      }),
    });
    const data = await res.json();

    removeTypingIndicator();

    if (!res.ok) {
      renderMessage("bot", `Maaf, terjadi kesalahan: ${data.error || "unknown"}`);
      return;
    }

    currentConversationId = data.conversation_id;
    renderMessage("bot", data.answer, data.time);
    loadHistory(); // refresh sidebar biar percakapan baru muncul
  } catch (err) {
    removeTypingIndicator();
    renderMessage("bot", "Maaf, tidak bisa terhubung ke server. Coba lagi.");
    console.error(err);
  } finally {
    sendButton.disabled = false;
  }
}

function currentTime() {
  const now = new Date();
  return `${String(now.getHours()).padStart(2, "0")}.${String(
    now.getMinutes()
  ).padStart(2, "0")}`;
}

// ===== Riwayat chat =====
async function loadHistory() {
  try {
    const res = await fetch("/api/conversations");
    const conversations = await res.json();

    historyList.innerHTML = "";
    conversations.forEach((conv) => {
      const item = document.createElement("div");
      item.className = "history-item";
      if (conv.id === currentConversationId) item.classList.add("active");
      item.textContent = conv.title;
      item.addEventListener("click", () => openConversation(conv.id));
      historyList.appendChild(item);
    });
  } catch (err) {
    console.error("Gagal memuat riwayat:", err);
  }
}

async function openConversation(conversationId) {
  try {
    const res = await fetch(`/api/conversations/${conversationId}/messages`);
    if (!res.ok) return;
    const messages = await res.json();

    currentConversationId = conversationId;
    chatArea.innerHTML = "";
    hideHeroIfNeeded();

    messages.forEach((m) => renderMessage(m.role, m.content, m.time));

    // tandai item aktif di sidebar
    document
      .querySelectorAll(".history-item")
      .forEach((el) => el.classList.remove("active"));
    loadHistory();
  } catch (err) {
    console.error("Gagal memuat percakapan:", err);
  }
}

function startNewConversation() {
  currentConversationId = null;
  chatArea.innerHTML = "";
  showHero();
  questionInput.value = "";
  questionInput.focus();
}

// ===== Event listeners =====
sendButton.addEventListener("click", () => sendMessage());

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

document.querySelectorAll(".suggestion").forEach((btn) => {
  btn.addEventListener("click", () => sendMessage(btn.textContent));
});

if (homeButton) {
  homeButton.addEventListener("click", startNewConversation);
}

if (historyButton) {
  historyButton.addEventListener("click", loadHistory);
}

// ===== Inisialisasi saat halaman dimuat =====
document.addEventListener("DOMContentLoaded", () => {
  loadHistory();
});