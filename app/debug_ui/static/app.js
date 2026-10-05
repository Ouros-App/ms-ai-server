if (window.location.search) {
  window.history.replaceState(
    window.history.state,
    "",
    window.location.pathname + window.location.hash,
  );
}

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
let idSequence = 0;

function createId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  }
  idSequence += 1;
  return "chat-" + Date.now().toString(36) + "-" + idSequence.toString(36);
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
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function markdown(value) {
  const lines = String(value ?? "").split("\n");
  const blocks = [];
  const isBlank = (line) => line.trim().length === 0;
  const fenceOf = (line) => {
    const trimmed = line.trimStart();
    const character = trimmed[0];
    if (character !== String.fromCharCode(96) && character !== "~") return null;
    let length = 0;
    while (trimmed[length] === character) length += 1;
    if (length < 3) return null;
    const info = trimmed.slice(length);
    if (character === String.fromCharCode(96) && info.includes(character)) return null;
    return { character, length, info };
  };
  const isFence = (line) => fenceOf(line) !== null;
  const closesFence = (line, opening) => {
    const trimmed = line.trimStart();
    let length = 0;
    while (trimmed[length] === opening.character) length += 1;
    return length >= opening.length && trimmed.slice(length).trim().length === 0;
  };
  const headingLevel = (line) => {
    let count = 0;
    while (line[count] === "#" && count < 3) count += 1;
    return count > 0 && line[count] === " " ? count : 0;
  };
  const listKind = (line) => {
    const trimmed = line.trimStart();
    if (trimmed.length > 2 && (trimmed.startsWith("- ") || trimmed.startsWith("* ") || trimmed.startsWith("+ "))) return "ul";
    let cursor = 0;
    while (cursor < trimmed.length && trimmed[cursor] >= "0" && trimmed[cursor] <= "9") cursor += 1;
    return cursor > 0 && (trimmed[cursor] === "." || trimmed[cursor] === ")")
      && trimmed[cursor + 1] === " " ? "ol" : "";
  };
  const renderInline = (text) => {
    let html = "";
    let cursor = 0;
    while (cursor < text.length) {
      const character = text[cursor];
      if (character === String.fromCharCode(96)) {
        const end = text.indexOf(character, cursor + 1);
        if (end > cursor + 1) {
          html += "<code>" + escapeText(text.slice(cursor + 1, end)) + "</code>";
          cursor = end + 1;
          continue;
        }
      }
      if (text.startsWith("**", cursor) || text.startsWith("__", cursor)) {
        const marker = text.slice(cursor, cursor + 2);
        const end = text.indexOf(marker, cursor + 2);
        if (end > cursor + 2) {
          html += "<strong>" + escapeText(text.slice(cursor + 2, end)) + "</strong>";
          cursor = end + 2;
          continue;
        }
      }
      if (character === "*") {
        const end = text.indexOf("*", cursor + 1);
        if (end > cursor + 1) {
          html += "<em>" + escapeText(text.slice(cursor + 1, end)) + "</em>";
          cursor = end + 1;
          continue;
        }
      }
      if (character === "[") {
        const labelEnd = text.indexOf("](", cursor + 1);
        const urlEnd = labelEnd >= 0 ? text.indexOf(")", labelEnd + 2) : -1;
        if (labelEnd > cursor + 1 && urlEnd > labelEnd + 2) {
          const label = text.slice(cursor + 1, labelEnd);
          const url = text.slice(labelEnd + 2, urlEnd);
          if ((url.startsWith("https://") || url.startsWith("http://")) && !url.includes(" ")) {
            html += '<a href="' + escapeText(url) + '" target="_blank" rel="noopener noreferrer">' + escapeText(label) + "</a>";
            cursor = urlEnd + 1;
            continue;
          }
        }
      }
      html += escapeText(character);
      cursor += 1;
    }
    return html;
  };

  let index = 0;
  while (index < lines.length) {
    if (isBlank(lines[index])) {
      index += 1;
      continue;
    }
    const openingFence = fenceOf(lines[index]);
    if (openingFence) {
      const language = openingFence.info.trim();
      const code = [];
      index += 1;
      while (index < lines.length && !closesFence(lines[index], openingFence)) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      const validLanguage = language.length > 0 && language.length <= 24
        && Array.from(language).every((char) =>
          (char >= "a" && char <= "z") || (char >= "A" && char <= "Z")
          || (char >= "0" && char <= "9") || char === "+" || char === "-" || char === "_");
      const languageClass = validLanguage ? ' class="language-' + escapeText(language) + '"' : "";
      blocks.push("<pre><code" + languageClass + ">" + escapeText(code.join("\n")) + "</code></pre>");
      continue;
    }
    const level = headingLevel(lines[index]);
    if (level) {
      blocks.push("<h" + level + ">" + renderInline(lines[index].slice(level + 1)) + "</h" + level + ">");
      index += 1;
      continue;
    }
    if (lines[index].trimStart().startsWith(">")) {
      const quote = [];
      while (index < lines.length && lines[index].trimStart().startsWith(">")) {
        let line = lines[index].trimStart().slice(1);
        if (line.startsWith(" ")) line = line.slice(1);
        quote.push(line);
        index += 1;
      }
      blocks.push("<blockquote>" + renderInline(quote.join(" ")) + "</blockquote>");
      continue;
    }
    const kind = listKind(lines[index]);
    if (kind) {
      const items = [];
      while (index < lines.length && listKind(lines[index]) === kind) {
        const line = lines[index].trimStart();
        let itemStart = 0;
        if (kind === "ul") itemStart = 2;
        else {
          while (itemStart < line.length && line[itemStart] >= "0" && line[itemStart] <= "9") itemStart += 1;
          itemStart += 2;
        }
        items.push("<li>" + renderInline(line.slice(itemStart)) + "</li>");
        index += 1;
      }
      blocks.push("<" + kind + ">" + items.join("") + "</" + kind + ">");
      continue;
    }
    const paragraph = [];
    while (index < lines.length && !isBlank(lines[index]) && !isFence(lines[index])
      && !headingLevel(lines[index]) && !lines[index].trimStart().startsWith(">")
      && !listKind(lines[index])) {
      paragraph.push(lines[index]);
      index += 1;
    }
    blocks.push("<p>" + renderInline(paragraph.join("\n")).replaceAll("\n", "<br>") + "</p>");
  }
  return blocks.join("");
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
