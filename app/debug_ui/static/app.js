const $ = (selector) => document.querySelector(selector);

const loginScreen = $("#login-screen");
const appShell = $("#app-shell");
const loginForm = $("#login-form");
const loginError = $("#login-error");
const loginButton = $("#login-button");
const composer = $("#composer");
const input = $("#message-input");
const sendButton = $("#send-button");
const messages = $("#messages");
const conversationList = $("#conversation-list");
const conversationTitle = $("#conversation-title");
const debugContent = $("#debug-content");
const debugPanel = $("#debug-panel");
const STORAGE_PREFIX = "ouros-ai-debug-conversations-v1";

let session = null;
let conversations = [];
let currentId = null;
let busy = false;

marked.setOptions({ gfm: true, breaks: true });

function storageKey() {
  if (!session?.user_id) return null;
  return `${STORAGE_PREFIX}:${session.account_type}:${session.user_id}`;
}

function loadConversations() {
  const key = storageKey();
  if (!key) return [];
  try {
    const parsed = JSON.parse(localStorage.getItem(key) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveConversations() {
  const key = storageKey();
  if (!key) return;
  localStorage.setItem(key, JSON.stringify(conversations.slice(0, 40)));
}

function restoreConversations() {
  localStorage.removeItem(STORAGE_PREFIX);
  conversations = loadConversations();
  currentId = conversations[0]?.id || null;
}

function newConversation() {
  const conversation = {
    id: crypto.randomUUID(),
    title: "Nova conversa",
    messages: [],
    latestDebug: null,
    createdAt: Date.now(),
  };
  conversations.unshift(conversation);
  currentId = conversation.id;
  saveConversations();
  renderAll();
  input.focus();
}

function currentConversation() {
  return conversations.find((item) => item.id === currentId) || null;
}

function escapeText(value) {
  const node = document.createElement("div");
  node.textContent = value ?? "";
  return node.innerHTML;
}

function markdown(value) {
  const source = String(value ?? "");
  return DOMPurify.sanitize(marked.parse(source), {
    USE_PROFILES: { html: true },
  });
}

function highlight(container) {
  container.querySelectorAll("pre code").forEach((block) => hljs.highlightElement(block));
}

function renderMessages() {
  const conversation = currentConversation();
  messages.innerHTML = "";
  if (!conversation?.messages.length) {
    messages.innerHTML = `
      <div class="empty-state">
        <span class="brand-mark">◒</span>
        <h3>Console de conversa</h3>
        <p>Mesma pipeline do produto, com o capô aberto.</p>
      </div>`;
    return;
  }

  for (const item of conversation.messages) {
    const row = document.createElement("div");
    row.className = `message-row ${item.role}`;
    const bubble = document.createElement("div");
    bubble.className = "message";
    const body = document.createElement("div");
    body.className = "markdown";
    body.innerHTML = markdown(item.content);
    bubble.appendChild(body);

    if (item.role === "assistant" && item.meta) {
      const meta = document.createElement("div");
      meta.className = "message-meta";
      const agents = item.meta.agents?.length ? item.meta.agents.join(" → ") : "sem agente";
      meta.textContent = `${item.meta.duration_ms ?? "?"} ms · ${agents}`;
      bubble.appendChild(meta);
    }

    row.appendChild(bubble);
    messages.appendChild(row);
    highlight(body);
  }
  messages.scrollTop = messages.scrollHeight;
}

function renderConversations() {
  conversationList.innerHTML = "";
  for (const conversation of conversations) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "conversation-item" + (conversation.id === currentId ? " active" : "");
    button.textContent = conversation.title || "Conversa";
    button.addEventListener("click", () => {
      currentId = conversation.id;
      renderAll();
    });
    conversationList.appendChild(button);
  }
}

function renderDebug(data) {
  if (!data) {
    debugContent.innerHTML = '<div class="debug-empty">Nenhum trace para esta conversa ainda.</div>';
    return;
  }

  const chipSet = (title, values, kind = "") => `
    <section class="debug-section">
      <h3>${escapeText(title)}</h3>
      <div class="chips">${(values || []).map((value) =>
        `<span class="chip ${kind}">${escapeText(value)}</span>`
      ).join("") || '<span class="chip">nenhum</span>'}</div>
    </section>`;

  const agents = (data.specialist_results || []).map((item) => `
    <article class="agent-card">
      <strong>${escapeText(item.agent || "agent")}</strong>
      <span class="chip ${item.status === "ok" ? "ok" : ""}">${escapeText(item.status || "?")}</span>
      <pre class="debug-json">${escapeText(JSON.stringify(item, null, 2))}</pre>
    </article>`).join("");

  const trace = (data.trace || []).map((event) => {
    const payload = { ...event };
    delete payload.t_ms;
    delete payload.event;
    return `
      <article class="trace-event">
        <div class="trace-line">
          <span class="trace-time">+${escapeText(event.t_ms)}ms</span>
          <span class="trace-name">${escapeText(event.event)}</span>
        </div>
        ${Object.keys(payload).length ? `<pre class="debug-json">${escapeText(JSON.stringify(payload, null, 2))}</pre>` : ""}
      </article>`;
  }).join("");

  debugContent.innerHTML = `
    <section class="debug-section">
      <h3>Resumo</h3>
      <div class="chips">
        <span class="chip gold">${escapeText(data.duration_ms)} ms</span>
        <span class="chip">${escapeText(data.guardrail?.allowed === false ? "bloqueado" : "guardrail ok")}</span>
      </div>
    </section>
    ${chipSet("Rotas", data.routes, "gold")}
    ${chipSet("Agentes", data.agents, "")}
    ${chipSet("Tools", data.tools, "")}
    <section class="debug-section">
      <h3>Respostas internas</h3>
      ${agents || '<div class="debug-empty">Nenhum especialista produziu resultado.</div>'}
    </section>
    <section class="debug-section">
      <h3>Logs da requisição</h3>
      ${trace || '<div class="debug-empty">Sem eventos.</div>'}
    </section>`;
}

function renderAll() {
  const conversation = currentConversation();
  conversationTitle.textContent = conversation?.title || "Nova conversa";
  renderConversations();
  renderMessages();
  renderDebug(conversation?.latestDebug || null);
}

function setLoading(active) {
  busy = active;
  sendButton.disabled = active;
  input.disabled = active;
  const existing = $("#loading-row");
  if (existing) existing.remove();
  if (!active) return;

  const row = document.createElement("div");
  row.id = "loading-row";
  row.className = "message-row assistant";
  row.innerHTML = '<div class="message"><div class="loading-message"><span></span><span></span><span></span><em>processando</em></div></div>';
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  if (response.status === 204) return null;
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const error = new Error(body?.detail || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return body;
}

function showLogin() {
  session = null;
  conversations = [];
  currentId = null;
  loginScreen.hidden = false;
  appShell.hidden = true;
  $("#password").value = "";
}

function showApp() {
  restoreConversations();
  loginScreen.hidden = true;
  appShell.hidden = false;
  $("#session-label").textContent = `${session.account_type} · DB #${session.user_id}`;
  if (!currentId) newConversation();
  else renderAll();
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginError.hidden = true;
  loginButton.disabled = true;
  loginButton.textContent = "Entrando…";
  try {
    session = await api("/debug/api/login", {
      method: "POST",
      body: JSON.stringify({
        email: $("#email").value,
        password: $("#password").value,
      }),
    });
    showApp();
  } catch (error) {
    loginError.textContent = error.message;
    loginError.hidden = false;
  } finally {
    loginButton.disabled = false;
    loginButton.textContent = "Entrar";
  }
});

$("#logout").addEventListener("click", async () => {
  try { await api("/debug/api/logout", { method: "POST" }); } catch {}
  showLogin();
});

$("#new-chat").addEventListener("click", newConversation);

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const text = input.value.trim();
  if (!text) return;
  let conversation = currentConversation();
  if (!conversation) {
    newConversation();
    conversation = currentConversation();
  }

  conversation.messages.push({ role: "user", content: text });
  if (conversation.title === "Nova conversa") {
    conversation.title = text.replace(/\s+/g, " ").slice(0, 48);
  }
  input.value = "";
  input.style.height = "auto";
  saveConversations();
  renderAll();
  setLoading(true);

  try {
    const data = await api("/debug/api/chat", {
      method: "POST",
      body: JSON.stringify({ message: text, thread_id: conversation.id }),
    });
    conversation.messages.push({
      role: "assistant",
      content: data.message,
      meta: data,
    });
    conversation.latestDebug = data;
    saveConversations();
  } catch (error) {
    if (error.status === 401) {
      showLogin();
      return;
    }
    conversation.messages.push({
      role: "assistant",
      content: `**Erro de debug:** ${error.message}`,
    });
    saveConversations();
  } finally {
    setLoading(false);
    renderAll();
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
});

$("#toggle-debug").addEventListener("click", () => debugPanel.classList.add("open"));
$("#close-debug").addEventListener("click", () => debugPanel.classList.remove("open"));

api("/debug/api/session")
  .then((restoredSession) => {
    session = restoredSession;
    showApp();
  })
  .catch(() => showLogin());
