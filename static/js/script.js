"use strict";

// ===== Elemen =====
const app = document.getElementById("app");
const mainPanel = document.getElementById("main");
const sidebar = document.getElementById("sidebar");
const sidebarOverlay = document.getElementById("sidebarOverlay");
const menuButton = document.getElementById("menuButton");
const sidebarClose = document.getElementById("sidebarClose");

const chatArea = document.getElementById("chatArea");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("sendButton");
const heroSection = document.getElementById("heroSection");
const suggestionSection = document.getElementById("suggestionSection");
const suggestionList = document.getElementById("suggestionList");
const historyList = document.getElementById("historyList");
const homeButton = document.getElementById("homeButton");
const historyButton = document.getElementById("historyButton");

let currentConversationId = null;

// ===== Tema terang / gelap =====
// Tema awal sudah dipasang oleh script kecil di <head> (anti-kilat). Di sini
// hanya: ganti tema lewat tombol, simpan pilihan, dan ikuti pengaturan
// sistem selama pengguna belum memilih sendiri.
const THEME_KEY = "chatbot-theme";
const systemDarkQuery = window.matchMedia("(prefers-color-scheme: dark)");

function getStoredTheme() {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return value === "dark" || value === "light" ? value : null;
  } catch (err) {
    return null; // mis. mode privat / penyimpanan diblokir
  }
}

function saveTheme(theme) {
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch (err) {
    /* tidak masalah: tema tetap berlaku untuk sesi ini */
  }
}

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "dark"
    ? "dark"
    : "light";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);

  // Warna bar status browser HP mengikuti warna header
  const themeColor = document.querySelector('meta[name="theme-color"]');
  if (themeColor) themeColor.setAttribute("content", theme === "dark" ? "#211e30" : "#ffffff");

  // Tombol menunjukkan tema yang AKAN dipilih saat ditekan
  const target = theme === "dark" ? "terang" : "gelap";
  document.querySelectorAll(".theme-toggle").forEach((btn) => {
    btn.setAttribute("aria-label", `Ganti ke mode ${target}`);
    const label = btn.querySelector(".theme-toggle-label");
    if (label) label.textContent = `Mode ${target}`;
  });
}

function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  saveTheme(next);
}

// ===== Rekomendasi pertanyaan (acak setiap Home / halaman dimuat) =====
// Tambah / ubah pertanyaan di sini. Pastikan topiknya memang ada di
// knowledge base supaya chatbot bisa menjawabnya.
const SUGGESTION_POOL = [
  "Apa itu KITAS?",
  "Apa itu KITAP?",
  "Apa perbedaan KITAS dan KITAP?",
  "Berapa Pembayaran KITAP",
  "Apa saja syarat membuat KITAS?",
  "Bagaimana cara memperpanjang KITAS?",
  "Cara apply SKIM",
  "Apa itu VOA?",
  "Apa saja syarat VOA?",
  "Bagaimana cara memperpanjang VOA?",
  "Berapa biaya perpanjangan VOA?",
  "Apa itu paspor elektronik?",
  "Apa saja syarat membuat paspor elektronik?",
  "Apa itu izin tinggal?",
  "Apa saja jenis visa di Indonesia?",
  "Bagaimana cara memperpanjang visa kunjungan?",
  "Berapa lama proses pembuatan KITAS?",
  "What are the requirements for VOA?",
  "How do I extend my stay permit?",
];
const SUGGESTION_COUNT = 4;
let shownSuggestions = [];

function shuffle(items) {
  const copy = items.slice();
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function pickSuggestions() {
  // Utamakan pertanyaan yang BELUM tampil di set sebelumnya,
  // supaya tiap kali Home dibuka isinya terasa berganti.
  const fresh = SUGGESTION_POOL.filter((q) => !shownSuggestions.includes(q));
  const source = fresh.length >= SUGGESTION_COUNT ? fresh : SUGGESTION_POOL;
  return shuffle(source).slice(0, SUGGESTION_COUNT);
}

function renderSuggestions() {
  if (!suggestionList) return;

  shownSuggestions = pickSuggestions();
  suggestionList.innerHTML = "";
  shownSuggestions.forEach((question, index) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "suggestion";
    btn.textContent = question;
    btn.style.setProperty("--i", index); // jeda animasi muncul bertahap
    suggestionList.appendChild(btn);
  });
}

// ===== Sidebar: panel permanen (layar lebar) atau drawer (layar sempit) =====
// Breakpoint HARUS sama dengan breakpoint sidebar permanen di style.css
// (56.25em = 900px pada ukuran teks standar).
const wideLayout = window.matchMedia("(min-width: 56.25em)");

function isDrawerOpen() {
  return app.classList.contains("sidebar-open");
}

function openSidebar() {
  // Di layar lebar sidebar sudah permanen, tidak ada drawer.
  if (wideLayout.matches || isDrawerOpen()) return;

  app.classList.add("sidebar-open");
  menuButton.setAttribute("aria-expanded", "true");
  // Area chat di belakang overlay tidak boleh bisa difokus / diklik
  mainPanel.setAttribute("inert", "");
  sidebarClose.focus({ preventScroll: true });
}

function closeSidebar() {
  if (!isDrawerOpen()) return;

  app.classList.remove("sidebar-open");
  menuButton.setAttribute("aria-expanded", "false");
  mainPanel.removeAttribute("inert");
  // Kembalikan fokus ke tombol yang membuka drawer
  menuButton.focus({ preventScroll: true });
}

// Kalau layar tiba-tiba jadi lebar (rotasi / resize) saat drawer terbuka,
// reset status drawer supaya tidak "nyangkut".
function handleLayoutChange() {
  if (!wideLayout.matches) return;
  app.classList.remove("sidebar-open");
  menuButton.setAttribute("aria-expanded", "false");
  mainPanel.removeAttribute("inert");
}

// ===== Scroll area pesan =====
function isNearBottom() {
  return chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight < 80;
}

function scrollChatToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

// ===== Render pesan ke #chatArea =====
function renderMessage(role, content, time, { scroll = true } = {}) {
  // Dicek SEBELUM pesan baru ditambahkan: kalau pengguna sedang membaca
  // pesan lama (scroll ke atas), layar tidak ditarik paksa ke bawah.
  // Pesan milik pengguna sendiri selalu diikuti.
  const shouldFollow = scroll && (role === "user" || isNearBottom());

  const row = document.createElement("div");
  row.className = `message-row ${role === "user" ? "user" : "bot"}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "🧑" : "🤖";

  const bubble = document.createElement("div");
  bubble.className = "message";

  const contentEl = document.createElement("div");
  contentEl.className = "message-content";
  if (role === "user") {
    // Teks pengguna ditampilkan apa adanya (bukan HTML)
    contentEl.textContent = content;
  } else if (typeof marked !== "undefined") {
    // marked.js untuk render markdown dari jawaban bot (list, bold, dll)
    contentEl.innerHTML = marked.parse(content);
  } else {
    contentEl.textContent = content;
  }

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
  if (shouldFollow) scrollChatToBottom();
}

function showTypingIndicator() {
  const shouldFollow = isNearBottom();

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
  if (shouldFollow) scrollChatToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

// "Belum ada pesan" = hero + rekomendasi terlihat, class is-empty aktif
// (CSS memakai class ini untuk menengahkan hero + input).
function hideHeroIfNeeded() {
  if (heroSection) heroSection.classList.add("hidden");
  if (suggestionSection) suggestionSection.classList.add("hidden");
  mainPanel.classList.remove("is-empty");
}

function showHero() {
  if (heroSection) heroSection.classList.remove("hidden");
  if (suggestionSection) suggestionSection.classList.remove("hidden");
  mainPanel.classList.add("is-empty");
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
      // <button> (bukan <div>) supaya bisa dipilih lewat keyboard / screen reader
      const item = document.createElement("button");
      item.type = "button";
      item.className = "history-item";
      if (conv.id === currentConversationId) {
        item.classList.add("active");
        item.setAttribute("aria-current", "true");
      }
      item.textContent = conv.title;
      item.title = conv.title; // judul penuh saat di-hover
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

    messages.forEach((m) =>
      renderMessage(m.role, m.content, m.time, { scroll: false })
    );
    scrollChatToBottom();

    // tandai item aktif di sidebar
    document
      .querySelectorAll(".history-item")
      .forEach((el) => el.classList.remove("active"));
    loadHistory();

    // Di layar sempit, drawer menutup otomatis setelah memilih percakapan
    closeSidebar();
  } catch (err) {
    console.error("Gagal memuat percakapan:", err);
  }
}

function startNewConversation() {
  currentConversationId = null;
  chatArea.innerHTML = "";
  showHero();
  renderSuggestions();
  questionInput.value = "";
  closeSidebar();
  // Di HP, fokus otomatis akan memunculkan keyboard; hanya dilakukan di layar lebar
  if (wideLayout.matches) questionInput.focus();
}

// ===== Tinggi layar yang mengikuti keyboard virtual HP =====
// Chrome Android sudah ditangani meta viewport (interactive-widget).
// iOS Safari tidak menciutkan layout saat keyboard muncul, jadi tinggi
// aplikasi disamakan dengan viewport visual supaya kolom input tetap
// terlihat. Dilewati saat pengguna sedang pinch-zoom.
function syncAppHeight() {
  const vv = window.visualViewport;
  if (!vv || vv.scale > 1.01) return;

  const wasNearBottom = isNearBottom();
  document.documentElement.style.setProperty(
    "--app-height",
    `${Math.round(vv.height)}px`
  );
  if (vv.offsetTop > 0) window.scrollTo(0, 0);
  // Jaga pesan terbaru tetap terlihat saat keyboard muncul / hilang
  if (wasNearBottom) scrollChatToBottom();
}

// ===== Event listeners (dipasang SEKALI saat halaman dimuat) =====
sendButton.addEventListener("click", () => sendMessage());

questionInput.addEventListener("keydown", (e) => {
  // isComposing: jangan kirim saat Enter dipakai untuk memilih saran keyboard HP
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    sendMessage();
  }
});

// Satu listener untuk semua tombol rekomendasi (tombolnya dibuat ulang setiap Home)
if (suggestionList) {
  suggestionList.addEventListener("click", (e) => {
    const btn = e.target.closest(".suggestion");
    if (btn) sendMessage(btn.textContent);
  });
}

document.querySelectorAll(".theme-toggle").forEach((btn) => {
  btn.addEventListener("click", toggleTheme);
});

// Ikuti pengaturan sistem selama pengguna belum memilih tema sendiri
systemDarkQuery.addEventListener("change", (e) => {
  if (!getStoredTheme()) applyTheme(e.matches ? "dark" : "light");
});

if (homeButton) {
  homeButton.addEventListener("click", startNewConversation);
}

if (historyButton) {
  historyButton.addEventListener("click", loadHistory);
}

// Drawer: tombol menu, tombol tutup, klik overlay, tombol Escape
menuButton.addEventListener("click", () => {
  if (isDrawerOpen()) closeSidebar();
  else openSidebar();
});
sidebarClose.addEventListener("click", closeSidebar);
sidebarOverlay.addEventListener("click", closeSidebar);

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && isDrawerOpen()) {
    e.preventDefault();
    closeSidebar();
  }
});

wideLayout.addEventListener("change", handleLayoutChange);

if (window.visualViewport) {
  window.visualViewport.addEventListener("resize", syncAppHeight);
  window.visualViewport.addEventListener("scroll", syncAppHeight);
  syncAppHeight();
}

// ===== Inisialisasi saat halaman dimuat =====
applyTheme(currentTheme()); // sinkronkan label tombol + warna bar status
renderSuggestions();

document.addEventListener("DOMContentLoaded", () => {
  loadHistory();
});