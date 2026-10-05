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
const sidebar = $("#sidebar");
const mobileScrim = $("#mobile-scrim");
const STORAGE_PREFIX = "ouros-ai-debug-conversations-v1";

let session = null;
let conversations = [];
let currentId = null;
let busy = false;

function createId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 12);
}

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
  const persistable = conversations.slice(0, 40);
  try {
    localStorage.setItem(key, JSON.stringify(persistable));
  } catch {
    const compact = persistable.map((conversation) => ({
      ...conversation,
      messages: conversation.messages.map(({ visualizations, ...message }) => message),
    }));
    try { localStorage.setItem(key, JSON.stringify(compact)); } catch {}
  }
}

async function restoreConversationHistory(conversation) {
  if (!conversation?.serverBacked || conversation.historyLoaded) return;
  try {
    const history = await api(
      `/debug/api/conversations/${encodeURIComponent(conversation.id)}`
    );
    conversation.messages = history.messages || [];
    conversation.historyLoaded = true;
    saveConversations();
  } catch (error) {
    if (error.status === 401) showLogin();
  }
}

async function restoreConversations() {
  localStorage.removeItem(STORAGE_PREFIX);
  const cached = loadConversations();
  conversations = cached;
  try {
    const saved = await api("/debug/api/conversations");
    const serverConversations = (saved.conversations || []).map((item) => {
      const existing = cached.find((conversation) => conversation.id === item.id);
      return {
        id: item.id,
        title: item.title,
        messages: existing?.messages || [],
        latestDebug: existing?.latestDebug || null,
        createdAt: existing?.createdAt || Date.now(),
        serverBacked: true,
        historyLoaded: false,
      };
    });
    const serverIds = new Set(serverConversations.map(({ id }) => id));
    const localDrafts = cached.filter(({ id }) => !serverIds.has(id));
    conversations = [...serverConversations, ...localDrafts].slice(0, 40);
  } catch {}
  currentId = conversations[0]?.id || null;
  await restoreConversationHistory(currentConversation());
}

function newConversation() {
  const conversation = {
    id: createId(),
    title: "Nova conversa",
    messages: [],
    latestDebug: null,
    createdAt: Date.now(),
    serverBacked: false,
    historyLoaded: true,
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
  const codeBlocks = [];
  const marker = String.fromCharCode(1);
  const source = String(value ?? "").replace(/```([^\n]*)\n([\s\S]*?)```/g, (_match, language, code) => {
    const placeholder = marker + "CODE" + codeBlocks.length + marker;
    const safeLanguage = /^[\w+-]{1,24}$/.test(language.trim())
      ? " class=\"language-" + language.trim() + "\""
      : "";
    codeBlocks.push("<pre><code" + safeLanguage + ">" + escapeText(code.replace(/\n$/, "")) + "</code></pre>");
    return placeholder;
  });
  const inline = (text) => {
    let html = escapeText(text);
    const codeSpans = [];
    html = html.replace(/`([^`]+)`/g, (_match, code) => {
      const placeholder = marker + "INLINE" + codeSpans.length + marker;
      codeSpans.push("<code>" + code + "</code>");
      return placeholder;
    });
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/__(.+?)__/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>");
    return html.replace(new RegExp(marker + "INLINE(\\d+)" + marker, "g"),
      (_match, index) => codeSpans[Number(index)] || "");
  };
  const blocks = source.split(/\n\s*\n/).map((block) => block.trim()).filter(Boolean);
  return blocks.map((block) => {
    const codeMatch = block.match(new RegExp("^" + marker + "CODE(\\d+)" + marker + "$"));
    if (codeMatch) return codeBlocks[Number(codeMatch[1])] || "";
    const heading = block.match(/^(#{1,3})\s+([\s\S]*)$/);
    if (heading) {
      const level = heading[1].length;
      return "<h" + level + ">" + inline(heading[2]) + "</h" + level + ">";
    }
    if (/^>\s?/.test(block)) {
      return "<blockquote>" + inline(block.replace(/^>\s?/gm, "").replace(/\n/g, " ")) + "</blockquote>";
    }
    const lines = block.split("\n");
    if (lines.every((line) => /^\s*[-*+]\s+/.test(line))) {
      return "<ul>" + lines.map((line) => "<li>" + inline(line.replace(/^\s*[-*+]\s+/, "")) + "</li>").join("") + "</ul>";
    }
    if (lines.every((line) => /^\s*\d+[.)]\s+/.test(line))) {
      return "<ol>" + lines.map((line) => "<li>" + inline(line.replace(/^\s*\d+[.)]\s+/, "")) + "</li>").join("") + "</ol>";
    }
    return "<p>" + inline(block).replace(/\n/g, "<br>") + "</p>";
  }).join("");
}

window.addEventListener("message", (event) => {
  const message = event.data;
  if (
    event.origin !== "null" ||
    !message ||
    typeof message !== "object" ||
    message.type !== "ouros-chart-resize" ||
    typeof message.height !== "number" ||
    !Number.isFinite(message.height)
  ) return;

  const frame = [...document.querySelectorAll("iframe.visualization-frame")]
    .find((item) => item.contentWindow === event.source);
  if (!frame) return;

  const height = Math.min(1200, Math.max(260, Math.ceil(message.height)));
  frame.style.height = `${height}px`;
});

function appendVisualizations(container, visualizations, variant = "debug") {
  if (!Array.isArray(visualizations) || !visualizations.length) return;
  const section = document.createElement("section");
  section.className = `${variant}-visualizations`;
  for (const dashboard of visualizations) {
    for (const chart of dashboard.charts || []) {
      const card = document.createElement("article");
      card.className = "visualization-card";

      const frame = document.createElement("iframe");
      frame.className = "visualization-frame";
      frame.title = chart.title || dashboard.title || "Gráfico";
      frame.name = "visualization-" + createId();
      frame.setAttribute("sandbox", "allow-scripts");
      frame.referrerPolicy = "no-referrer";

      const form = document.createElement("form");
      form.method = "post";
      form.action = "/debug/api/visualization";
      form.target = frame.name;
      form.hidden = true;
      const payload = document.createElement("input");
      payload.type = "hidden";
      payload.name = "html";
      payload.value = chart.html;
      form.appendChild(payload);

      let submitted = false;
      frame.addEventListener("load", () => {
        if (!submitted) {
          submitted = true;
          document.body.appendChild(form);
          form.submit();
          return;
        }
        form.remove();
      });

      card.appendChild(frame);
      section.appendChild(card);
    }
  }
  if (!section.querySelector(".visualization-card")) return;
  container.appendChild(section);
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

    if (item.role === "assistant") {
      appendVisualizations(bubble, item.visualizations, "message");
    }

    if (item.role === "assistant" && item.meta) {
      const meta = document.createElement("div");
      meta.className = "message-meta";
      const agents = item.meta.agents?.length ? item.meta.agents.join(" → ") : "sem agente";
      meta.textContent = `${item.meta.duration_ms ?? "?"} ms · ${agents}`;
      bubble.appendChild(meta);
    }

    row.appendChild(bubble);
    messages.appendChild(row);
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
    button.addEventListener("click", async () => {
      currentId = conversation.id;
      closeMobilePanels();
      await restoreConversationHistory(conversation);
      renderAll();
    });
    conversationList.appendChild(button);
  }
}

function renderDebug(data, visualizations = []) {
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

  const agents = (data.specialist_results || []).map((item) => {
    const missing = (item.missing_data || []).map((value) =>
      `<span class="chip gold">${escapeText(value)}</span>`
    ).join("");
    return `
      <article class="agent-card">
        <div class="agent-card-head">
          <strong>${escapeText(item.agent || "agent")}</strong>
          <span class="chip ${item.status === "ok" ? "ok" : ""}">${escapeText(item.status || "?")}</span>
        </div>
        <div class="chips">
          <span class="chip">${escapeText(item.fact_count || 0)} fatos</span>
          <span class="chip">${escapeText(item.recommendation_count || 0)} recomendações</span>
          <span class="chip">${escapeText(item.source_count || 0)} fontes</span>
        </div>
        ${missing ? `<div class="agent-missing"><small>Aguardando</small><div class="chips">${missing}</div></div>` : ""}
      </article>`;
  }).join("");

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
        ${data.route_source ? `<span class="chip">${escapeText("rota: " + data.route_source)}</span>` : ""}
      </div>
    </section>
    ${chipSet("Rotas", data.routes, "gold")}
    ${chipSet("Pendências", data.pending_routes, "")}
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
  appendVisualizations(debugContent, visualizations);
}

function renderAll() {
  const conversation = currentConversation();
  conversationTitle.textContent = conversation?.title || "Nova conversa";
  renderConversations();
  renderMessages();
  const latestAssistant = [...(conversation?.messages || [])]
    .reverse()
    .find((item) => item.role === "assistant");
  renderDebug(conversation?.latestDebug || null, latestAssistant?.visualizations);
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

async function showApp() {
  loginScreen.hidden = true;
  appShell.hidden = false;
  $("#session-label").textContent = `${session.account_type} · DB #${session.user_id}`;
  await restoreConversations();
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
    await showApp();
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

$("#new-chat").addEventListener("click", () => {
  closeMobilePanels();
  newConversation();
});

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
      meta: { agents: data.agents, duration_ms: data.duration_ms },
      visualizations: data.visualizations || [],
    });
    conversation.serverBacked = true;
    conversation.historyLoaded = true;
    const { visualizations, ...debugData } = data;
    conversation.latestDebug = debugData;
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

function closeMobilePanels() {
  sidebar.classList.remove("open");
  debugPanel.classList.remove("open");
  mobileScrim.hidden = true;
}

$("#toggle-sidebar").addEventListener("click", () => {
  const opening = !sidebar.classList.contains("open");
  closeMobilePanels();
  if (opening) {
    sidebar.classList.add("open");
    mobileScrim.hidden = false;
  }
});
$("#toggle-debug").addEventListener("click", () => {
  const opening = !debugPanel.classList.contains("open");
  closeMobilePanels();
  if (opening) {
    debugPanel.classList.add("open");
    mobileScrim.hidden = false;
  }
});
$("#close-debug").addEventListener("click", closeMobilePanels);
mobileScrim.addEventListener("click", closeMobilePanels);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeMobilePanels();
});

api("/debug/api/session")
  .then(async (restoredSession) => {
    session = restoredSession;
    await showApp();
  })
  .catch(() => showLogin());
