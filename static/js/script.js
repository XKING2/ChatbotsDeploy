
"use strict";

/* =====================================================
   ELEMENT HTML
===================================================== */
const $ = (selector) => document.querySelector(selector);

const app = $("#app");
const main = $("#main");
const sidebar = $("#sidebar");
const sidebarOverlay = $("#sidebarOverlay");
const sidebarClose = $("#sidebarClose");
const menuButton = $("#menuButton");
const chatArea = $("#chatArea");
const chatForm = $("#chatForm");
const questionInput = $("#question");
const sendButton = $("#sendButton");
const suggestionList = $("#suggestionList");
const historyList = $("#historyList");

const LANGUAGE_KEY = "chatbot-language";
const THEME_KEY = "chatbot-theme";

let currentConversationId = null;
let currentLanguage = "id";
let isSending = false;

/* =====================================================
   TRANSLATIONS
===================================================== */
const translations = {
    id: {
        home: "Home",
        history: "Riwayat Chat",
        greeting: "HALO PENGUNJUNG",
        subtitle: "Apa yang bisa saya bantu hari ini?",
        suggestions: "Rekomendasi Pertanyaan",
        placeholder: "Masukkan pertanyaan Anda di sini",
        send: "Kirim pesan",
        themeDark: "Mode gelap",
        themeLight: "Mode terang",
        connectionError: "Maaf, tidak bisa terhubung ke server. Coba lagi.",
        responseError: "Maaf, terjadi kesalahan saat memproses pertanyaan.",
        emptyAnswer: "Maaf, belum ada jawaban yang bisa ditampilkan.",
        loading: "Sedang memproses...",
        openMenu: "Buka menu",
        closeMenu: "Tutup menu"
    },

    en: {
        home: "Home",
        history: "Chat History",
        greeting: "HELLO, VISITOR",
        subtitle: "How can I help you today?",
        suggestions: "Suggested Questions",
        placeholder: "Type your question here",
        send: "Send message",
        themeDark: "Dark mode",
        themeLight: "Light mode",
        connectionError: "Unable to connect to the server. Please try again.",
        responseError: "Sorry, an error occurred while processing your question.",
        emptyAnswer: "Sorry, no answer is available yet.",
        loading: "Processing...",
        openMenu: "Open menu",
        closeMenu: "Close menu"
    }
};

function t(key) {
    return translations[currentLanguage]?.[key]
        ?? translations.id[key]
        ?? key;
}

/* =====================================================
   PREFERENSI BAHASA
===================================================== */
function loadLanguage() {
    try {
        currentLanguage =
            localStorage.getItem(LANGUAGE_KEY) === "en" ? "en" : "id";
    } catch (_) {
        currentLanguage = "id";
    }
}

const suggestionPool = {
    id: [
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
        "Berapa lama proses pembuatan KITAS?"
    ],

    en: [
        "What is a KITAS?",
        "What is a KITAP?",
        "What is the difference between KITAS and KITAP?",
        "What are the requirements for a KITAS?",
        "How do I extend my KITAS?",
        "What is a Visa on Arrival (VOA)?",
        "What are the requirements for a VOA?",
        "How do I extend my VOA?",
        "How much does a VOA extension cost?",
        "What is an electronic passport?",
        "What are the requirements for an electronic passport?",
        "What is a stay permit?",
        "What types of visas are available in Indonesia?",
        "How do I extend a visit visa?",
        "How long does KITAS processing take?"
    ]
};

function shuffle(items) {
    const result = [...items];

    for (let i = result.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [result[i], result[j]] = [result[j], result[i]];
    }

    return result;
}

function renderSuggestions() {
    if (!suggestionList) return;

    suggestionList.replaceChildren();

    shuffle(suggestionPool[currentLanguage])
        .slice(0, 4)
        .forEach((question) => {
            const button = document.createElement("button");

            button.type = "button";
            button.className = "suggestion";
            button.textContent = question;

            suggestionList.appendChild(button);
        });
}

function applyLanguage(language) {
    currentLanguage = language === "en" ? "en" : "id";

    document.documentElement.lang = currentLanguage;

    try {
        localStorage.setItem(LANGUAGE_KEY, currentLanguage);
    } catch (_) {}

    const idButton = $("#languageId");
    const enButton = $("#languageEn");

    if (idButton) {
        idButton.classList.toggle("active", currentLanguage === "id");
        idButton.setAttribute(
            "aria-pressed",
            String(currentLanguage === "id")
        );
    }

    if (enButton) {
        enButton.classList.toggle("active", currentLanguage === "en");
        enButton.setAttribute(
            "aria-pressed",
            String(currentLanguage === "en")
        );
    }

    const elements = {
        homeLabel: "home",
        historyTitle: "history",
        heroHeading: "greeting",
        heroSubtitle: "subtitle",
        suggestionTitle: "suggestions"
    };

    Object.entries(elements).forEach(([id, key]) => {
        const element = document.getElementById(id);

        if (element) {
            element.textContent = t(key);
        }
    });

    if (questionInput) {
        questionInput.placeholder = t("placeholder");
        questionInput.setAttribute("aria-label", t("placeholder"));
    }

    if (sendButton) {
        sendButton.setAttribute("aria-label", t("send"));
    }

    if (menuButton) {
        menuButton.setAttribute("aria-label", t("openMenu"));
    }

    if (sidebarClose) {
        sidebarClose.setAttribute("aria-label", t("closeMenu"));
    }

    document.querySelectorAll(".theme-toggle").forEach((button) => {
        const label = button.querySelector(".theme-toggle-label");

        button.setAttribute(
            "aria-label",
            currentTheme() === "dark" ? t("themeLight") : t("themeDark")
        );

        if (label) {
            label.textContent = currentTheme() === "dark"
                ? t("themeLight")
                : t("themeDark");
        }
    });

    renderSuggestions();
}

$("#languageId")?.addEventListener("click", () => {
    if (currentLanguage !== "id") {
        applyLanguage("id");
    }
});

$("#languageEn")?.addEventListener("click", () => {
    if (currentLanguage !== "en") {
        applyLanguage("en");
    }
});

/* =====================================================
   DARK MODE
===================================================== */
function currentTheme() {
    return document.documentElement.dataset.theme === "dark"
        ? "dark"
        : "light";
}

function applyTheme(theme) {
    const selectedTheme = theme === "dark" ? "dark" : "light";

    document.documentElement.dataset.theme = selectedTheme;

    const meta = $('meta[name="theme-color"]');

    if (meta) {
        meta.content = selectedTheme === "dark" ? "#17161d" : "#ffffff";
    }

    document.querySelectorAll(".theme-toggle").forEach((button) => {
        const label = button.querySelector(".theme-toggle-label");

        button.setAttribute(
            "aria-label",
            selectedTheme === "dark" ? t("themeLight") : t("themeDark")
        );

        if (label) {
            label.textContent = selectedTheme === "dark"
                ? t("themeLight")
                : t("themeDark");
        }
    });
}

function toggleTheme() {
    const next = currentTheme() === "dark" ? "light" : "dark";

    applyTheme(next);

    try {
        localStorage.setItem(THEME_KEY, next);
    } catch (_) {}
}

document.querySelectorAll(".theme-toggle").forEach((button) => {
    button.addEventListener("click", toggleTheme);
});

/* =====================================================
   SIDEBAR RESPONSIF
===================================================== */
const wideLayout = window.matchMedia("(min-width: 901px)");

function openSidebar() {
    if (!app || wideLayout.matches) return;

    app.classList.add("sidebar-open");
    menuButton?.setAttribute("aria-expanded", "true");
}

function closeSidebar() {
    if (!app) return;

    app.classList.remove("sidebar-open");
    menuButton?.setAttribute("aria-expanded", "false");
}

menuButton?.addEventListener("click", () => {
    if (app?.classList.contains("sidebar-open")) {
        closeSidebar();
    } else {
        openSidebar();
    }
});

sidebarClose?.addEventListener("click", closeSidebar);
sidebarOverlay?.addEventListener("click", closeSidebar);

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeSidebar();
});

wideLayout.addEventListener("change", (event) => {
    if (event.matches) closeSidebar();
});

/* =====================================================
   TATA LETAK CHAT
===================================================== */
function updateChatLayout() {
    const hasMessages = Boolean(chatArea?.children.length);

    main?.classList.toggle("has-messages", hasMessages);
    main?.classList.toggle("is-empty", !hasMessages);
}

function scrollToBottom() {
    if (chatArea) {
        chatArea.scrollTop = chatArea.scrollHeight;
    }
}

function getCurrentTime() {
    return new Intl.DateTimeFormat(
        currentLanguage === "en" ? "en-US" : "id-ID",
        {
            hour: "2-digit",
            minute: "2-digit",
            hour12: false
        }
    ).format(new Date());
}

/* =====================================================
   RENDER PESAN
===================================================== */
function renderMessage(role, content, time = getCurrentTime()) {
    if (!chatArea) return;

    const row = document.createElement("div");
    row.className = `message-row ${role === "user" ? "user" : "bot"}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "🧑" : "🤖";
    avatar.setAttribute("aria-hidden", "true");

    const bubble = document.createElement("div");
    bubble.className = "message";

    const body = document.createElement("div");
    body.className = "message-content";

    if (role === "user") {
        body.textContent = content;
    } else if (typeof marked !== "undefined") {
        body.innerHTML = marked.parse(content || "");
    } else {
        body.textContent = content || "";
    }

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = time || getCurrentTime();

    bubble.append(body, meta);

    if (role === "user") {
        row.append(bubble, avatar);
    } else {
        row.append(avatar, bubble);
    }

    chatArea.appendChild(row);

    updateChatLayout();
    scrollToBottom();
}

function showTyping() {
    if (!chatArea || $("#typingIndicator")) return;

    const row = document.createElement("div");
    row.id = "typingIndicator";
    row.className = "message-row bot";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "🤖";

    const bubble = document.createElement("div");
    bubble.className = "message";

    const typing = document.createElement("div");
    typing.className = "typing";
    typing.setAttribute("aria-label", t("loading"));

    for (let i = 0; i < 3; i++) {
        typing.appendChild(document.createElement("span"));
    }

    bubble.appendChild(typing);
    row.append(avatar, bubble);
    chatArea.appendChild(row);

    updateChatLayout();
    scrollToBottom();
}

function removeTyping() {
    $("#typingIndicator")?.remove();
    updateChatLayout();
}

/* =====================================================
   KIRIM PESAN
===================================================== */
async function sendMessage(message) {
    const question = String(message ?? questionInput?.value ?? "").trim();

    if (!question || isSending) return;

    isSending = true;

    if (sendButton) sendButton.disabled = true;
    if (questionInput) questionInput.value = "";

    renderMessage("user", question);
    showTyping();
    closeSidebar();

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: question,
                conversation_id: currentConversationId,
                language: currentLanguage
            })
        });

        let data;

        try {
            data = await response.json();
        } catch (_) {
            data = {};
        }

        removeTyping();

        if (!response.ok) {
            renderMessage(
                "bot",
                data.error || t("responseError")
            );
            return;
        }

        if (data.conversation_id != null) {
            currentConversationId = data.conversation_id;
        }

        renderMessage(
            "bot",
            data.answer || data.response || t("emptyAnswer"),
            data.time || getCurrentTime()
        );

        loadHistory();
    } catch (error) {
        removeTyping();
        renderMessage("bot", t("connectionError"));
        console.error("Kesalahan saat mengirim pesan:", error);
    } finally {
        isSending = false;

        if (sendButton) sendButton.disabled = false;
    }
}

chatForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(questionInput?.value);
});

suggestionList?.addEventListener("click", (event) => {
    const button = event.target.closest(".suggestion");

    if (button) {
        sendMessage(button.textContent);
    }
});

/* =====================================================
   RIWAYAT CHAT
   Sesuaikan URL ini dengan route yang tersedia di app.py.
===================================================== */
async function loadHistory() {
    if (!historyList) return;

    try {
        const response = await fetch("/api/conversations");

        if (!response.ok) return;

        const result = await response.json();

        const conversations = Array.isArray(result)
            ? result
            : result.conversations || [];

        historyList.replaceChildren();

        conversations.forEach((conversation) => {
            const id = conversation.id ?? conversation.conversation_id;
            const title = conversation.title
                || conversation.name
                || conversation.last_message
                || "Percakapan";

            const button = document.createElement("button");

            button.type = "button";
            button.className = "history-item";
            button.textContent = title;
            button.title = title;

            if (String(id) === String(currentConversationId)) {
                button.classList.add("active");
            }

            button.addEventListener("click", () => openConversation(id));

            historyList.appendChild(button);
        });
    } catch (error) {
        console.warn("Riwayat chat tidak dapat dimuat:", error);
    }
}

async function openConversation(conversationId) {
    if (conversationId == null) return;

    try {
        const response = await fetch(
            `/api/conversations/${encodeURIComponent(conversationId)}/messages`
        );

        if (!response.ok) {
            console.warn("Endpoint pesan riwayat tidak tersedia.");
            return;
        }

        const result = await response.json();

        const messages = Array.isArray(result)
            ? result
            : result.messages || [];

        currentConversationId = conversationId;
        chatArea.replaceChildren();

        messages.forEach((message) => {
            const role =
                message.role === "user" || message.sender === "user"
                    ? "user"
                    : "bot";

            renderMessage(
                role,
                message.content ?? message.message ?? "",
                message.time ?? message.created_at ?? ""
            );
        });

        updateChatLayout();
        scrollToBottom();
        closeSidebar();
        loadHistory();
    } catch (error) {
        console.error("Gagal membuka percakapan:", error);
    }
}

/* =====================================================
   HOME / CHAT BARU
===================================================== */
function startNewChat() {
    currentConversationId = null;

    chatArea?.replaceChildren();

    if (questionInput) {
        questionInput.value = "";
    }

    removeTyping();
    updateChatLayout();
    renderSuggestions();
    closeSidebar();
    loadHistory();
}

$("#homeButton")?.addEventListener("click", startNewChat);

/* =====================================================
   VIEWPORT MOBILE
===================================================== */
function updateViewportHeight() {
    const height = window.visualViewport?.height || window.innerHeight;

    document.documentElement.style.setProperty(
        "--app-height",
        `${Math.round(height)}px`
    );
}

window.addEventListener("resize", updateViewportHeight);

window.visualViewport?.addEventListener(
    "resize",
    updateViewportHeight
);

/* =====================================================
   INISIALISASI
===================================================== */
function initializeChatbot() {
    loadLanguage();
    applyLanguage(currentLanguage);

    let theme = "light";

    try {
        const savedTheme = localStorage.getItem(THEME_KEY);

        if (savedTheme === "dark" || savedTheme === "light") {
            theme = savedTheme;
        } else {
            theme = window.matchMedia("(prefers-color-scheme: dark)").matches
                ? "dark"
                : "light";
        }
    } catch (_) {}

    applyTheme(theme);
    updateViewportHeight();
    updateChatLayout();
    loadHistory();
}

initializeChatbot();
