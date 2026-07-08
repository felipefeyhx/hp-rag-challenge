// HP RAG Chatbot - vanilla JS frontend.
// Talks to the same-origin FastAPI backend (no CORS, relative URLs).

const state = {
  sessionId: localStorage.getItem("session_id") || null,
};

// --------------------------------------------------------------------------- //
// API helpers                                                                 //
// --------------------------------------------------------------------------- //

async function apiGet(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`GET ${path} -> ${r.status}`);
  return r.json();
}

async function apiSend(method, path, body) {
  const opts = { method };
  if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${method} ${path} -> ${r.status}`);
  if (r.status === 204) return null;
  return r.json();
}

const api = {
  health: () => apiGet("/health"),
  listSessions: () => apiGet("/sessions"),
  createSession: () => apiSend("POST", "/sessions"),
  getSession: (id) => apiGet(`/sessions/${id}`),
  deleteSession: (id) => apiSend("DELETE", `/sessions/${id}`),
  sendMessage: (sessionId, message) =>
    apiSend("POST", "/chat", { session_id: sessionId, message }),
  streamMessage: (sessionId, message) =>
    fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ session_id: sessionId, message }),
    }),
};

// --------------------------------------------------------------------------- //
// Streaming helpers                                                           //
// --------------------------------------------------------------------------- //

/**
 * Parse a Server-Sent Events stream. Yields `{event, data}` objects where
 * `data` is the parsed JSON payload. Handles multi-line SSE blocks and
 * carriers left over between network chunks.
 */
async function* parseSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Each event ends with a blank line ("\n\n").
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);

      let event = "message";
      const dataLines = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length === 0) continue;
      try {
        yield { event, data: JSON.parse(dataLines.join("\n")) };
      } catch {
        // Non-JSON payload — skip.
      }
    }
  }
}

/**
 * Incremental splitter for a token stream that may contain <think>…</think>
 * blocks. Feed chunks in, get callbacks for reasoning tokens and answer
 * tokens as they resolve. Handles tags split across chunks by holding back
 * an ambiguous suffix in the buffer.
 */
function makeThinkStreamSplitter(onThinking, onAnswer) {
  const OPEN = "<think>";
  const CLOSE = "</think>";
  const OPEN_HOLD = OPEN.length - 1;    // 6 chars we might need to hold
  const CLOSE_HOLD = CLOSE.length - 1;  // 7 chars we might need to hold

  let mode = "answer";  // or "thinking"
  let buffer = "";

  const flush = (finalFlush) => {
    while (true) {
      if (mode === "answer") {
        const idx = buffer.indexOf(OPEN);
        if (idx !== -1) {
          if (idx > 0) onAnswer(buffer.slice(0, idx));
          buffer = buffer.slice(idx + OPEN.length);
          mode = "thinking";
          continue;
        }
        // No open tag yet. Emit everything except the last few chars that
        // could still start an open tag.
        const safeEnd = finalFlush ? buffer.length : Math.max(0, buffer.length - OPEN_HOLD);
        if (safeEnd > 0) {
          onAnswer(buffer.slice(0, safeEnd));
          buffer = buffer.slice(safeEnd);
        }
        return;
      } else {  // thinking
        const idx = buffer.indexOf(CLOSE);
        if (idx !== -1) {
          if (idx > 0) onThinking(buffer.slice(0, idx));
          buffer = buffer.slice(idx + CLOSE.length);
          mode = "answer";
          continue;
        }
        const safeEnd = finalFlush ? buffer.length : Math.max(0, buffer.length - CLOSE_HOLD);
        if (safeEnd > 0) {
          onThinking(buffer.slice(0, safeEnd));
          buffer = buffer.slice(safeEnd);
        }
        return;
      }
    }
  };

  return {
    feed(chunk) { buffer += chunk; flush(false); },
    end()       { flush(true); },
  };
}

// --------------------------------------------------------------------------- //
// DOM references                                                              //
// --------------------------------------------------------------------------- //

const el = {
  status: document.getElementById("status"),
  sessionList: document.getElementById("session-list"),
  messages: document.getElementById("messages"),
  emptyState: document.getElementById("empty-state"),
  form: document.getElementById("chat-form"),
  input: document.getElementById("chat-input"),
  sendBtn: document.getElementById("send-btn"),
  newChatBtn: document.getElementById("new-chat"),
};

// --------------------------------------------------------------------------- //
// Rendering                                                                   //
// --------------------------------------------------------------------------- //

function clearMessages() {
  el.messages.innerHTML = "";
}

function renderEmptyState() {
  clearMessages();
  const div = document.createElement("div");
  div.className = "empty-state";
  div.textContent = "Ask a question about the HP documents to get started.";
  el.messages.appendChild(div);
}

/**
 * Split an assistant reply into an optional reasoning trace and the
 * user-facing answer. Handles both closed <think>...</think> blocks and
 * unclosed ones (the model ran out of tokens mid-reasoning).
 */
function splitThinking(content) {
  if (!content) return { think: "", answer: "" };
  const closed = /<think>([\s\S]*?)<\/think>/gi;
  const parts = [];
  let lastIndex = 0;
  let m;
  while ((m = closed.exec(content)) !== null) {
    parts.push(m[1].trim());
    lastIndex = m.index + m[0].length;
  }
  let remainder = content.slice(lastIndex);

  // Handle an unclosed <think> (truncated model output).
  const openIdx = remainder.toLowerCase().indexOf("<think>");
  if (openIdx !== -1) {
    parts.push(remainder.slice(openIdx + "<think>".length).trim());
    remainder = remainder.slice(0, openIdx);
  }

  return { think: parts.join("\n\n").trim(), answer: remainder.trim() };
}

function renderThinkingBubble(text) {
  const details = document.createElement("details");
  details.className = "thinking";

  const summary = document.createElement("summary");
  summary.textContent = "💭 Thinking";
  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "thinking-body";
  body.textContent = text;
  details.appendChild(body);

  return details;
}

function renderMessage(role, content, sources) {
  // Remove empty-state placeholder if present.
  const empty = el.messages.querySelector(".empty-state");
  if (empty) empty.remove();

  const msg = document.createElement("div");
  msg.className = `msg ${role}`;

  let bubble;
  if (role === "assistant") {
    const { think, answer } = splitThinking(content);

    if (think) {
      msg.appendChild(renderThinkingBubble(think));
    }

    bubble = document.createElement("div");
    bubble.className = "bubble";
    // If reasoning consumed all tokens and no final answer was produced,
    // show a small placeholder so the bubble isn't blank.
    bubble.textContent = answer || (think ? "(no final answer produced — see reasoning above)" : content);
    msg.appendChild(bubble);
  } else {
    bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = content;
    msg.appendChild(bubble);
  }

  if (role === "assistant" && sources && sources.length > 0) {
    msg.appendChild(renderSources(sources));
  }

  el.messages.appendChild(msg);
  el.messages.scrollTop = el.messages.scrollHeight;
  return bubble;
}

function renderSources(sources) {
  const details = document.createElement("details");
  details.className = "sources";

  const summary = document.createElement("summary");
  summary.textContent = `Sources (${sources.length})`;
  details.appendChild(summary);

  sources.forEach((src, i) => {
    const item = document.createElement("div");
    item.className = "source-item";

    const head = document.createElement("div");
    head.className = "src-head";
    const scoreStr =
      typeof src.score === "number" ? ` · similarity ${src.score.toFixed(2)}` : "";
    head.textContent = `[${i + 1}] ${src.source}, p.${src.page}${scoreStr}`;
    item.appendChild(head);

    if (src.text) {
      const text = document.createElement("div");
      text.className = "src-text";
      text.textContent =
        src.text.length > 500 ? src.text.slice(0, 500) + "…" : src.text;
      item.appendChild(text);
    }

    details.appendChild(item);
  });

  return details;
}

async function renderSessionList() {
  let sessions = [];
  try {
    sessions = await api.listSessions();
  } catch (e) {
    return;
  }

  el.sessionList.innerHTML = "";
  sessions.forEach((s) => {
    const li = document.createElement("li");
    li.className = "session-item" + (s.id === state.sessionId ? " active" : "");

    const title = document.createElement("span");
    title.className = "session-title";
    title.textContent = s.title || "New chat";
    title.onclick = () => selectSession(s.id);
    li.appendChild(title);

    const del = document.createElement("button");
    del.className = "session-delete";
    del.textContent = "🗑";
    del.title = "Delete session";
    del.onclick = async (ev) => {
      ev.stopPropagation();
      await api.deleteSession(s.id);
      if (state.sessionId === s.id) {
        state.sessionId = null;
        localStorage.removeItem("session_id");
        renderEmptyState();
      }
      renderSessionList();
    };
    li.appendChild(del);

    el.sessionList.appendChild(li);
  });
}

// --------------------------------------------------------------------------- //
// Actions                                                                     //
// --------------------------------------------------------------------------- //

async function selectSession(id) {
  state.sessionId = id;
  localStorage.setItem("session_id", id);
  clearMessages();
  try {
    const sess = await api.getSession(id);
    const messages = sess.messages || [];
    if (messages.length === 0) {
      renderEmptyState();
    } else {
      messages.forEach((m) => renderMessage(m.role, m.content, m.sources));
    }
  } catch (e) {
    renderEmptyState();
  }
  renderSessionList();
}

async function startNewChat() {
  state.sessionId = null;
  localStorage.removeItem("session_id");
  renderEmptyState();
  renderSessionList();
  el.input.focus();
}

/**
 * Build an empty assistant message shell that we can stream tokens into.
 * Returns handles for progressively updating the thinking balloon and the
 * answer bubble.
 */
function beginStreamingAssistantMessage() {
  // Remove empty-state placeholder if present.
  const empty = el.messages.querySelector(".empty-state");
  if (empty) empty.remove();

  const msg = document.createElement("div");
  msg.className = "msg assistant";

  // Thinking balloon is created lazily on the first reasoning token.
  let thinkDetails = null;
  let thinkBody = null;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  msg.appendChild(bubble);

  // Show an animated typing indicator until the first token arrives, so the
  // bubble never looks like a blank/broken balloon while the model warms up.
  let typing = document.createElement("div");
  typing.className = "typing";
  typing.setAttribute("aria-label", "Assistant is typing");
  typing.innerHTML = "<span></span><span></span><span></span>";
  bubble.appendChild(typing);

  const clearTyping = () => {
    if (typing) {
      typing.remove();
      typing = null;
    }
  };

  el.messages.appendChild(msg);
  el.messages.scrollTop = el.messages.scrollHeight;

  const scrollToBottom = () => {
    el.messages.scrollTop = el.messages.scrollHeight;
  };

  return {
    root: msg,
    appendThinking(text) {
      clearTyping();
      if (!thinkDetails) {
        thinkDetails = document.createElement("details");
        thinkDetails.className = "thinking";
        thinkDetails.setAttribute("open", "");  // expanded while streaming
        const summary = document.createElement("summary");
        summary.textContent = "💭 Thinking…";
        thinkDetails.appendChild(summary);
        thinkBody = document.createElement("div");
        thinkBody.className = "thinking-body";
        thinkDetails.appendChild(thinkBody);
        msg.insertBefore(thinkDetails, bubble);
      }
      thinkBody.textContent += text;
      scrollToBottom();
    },
    appendAnswer(text) {
      clearTyping();
      bubble.textContent += text;
      scrollToBottom();
    },
    finalize(sources) {
      clearTyping();
      if (thinkDetails) {
        // Reasoning is done — flip label and collapse by default.
        thinkDetails.querySelector("summary").textContent = "💭 Thinking";
        thinkDetails.removeAttribute("open");
      }
      if (!bubble.textContent) {
        bubble.textContent = thinkDetails
          ? "(no final answer produced — see reasoning above)"
          : "(empty response)";
      }
      if (sources && sources.length > 0) {
        msg.appendChild(renderSources(sources));
      }
      scrollToBottom();
    },
    fail(message) {
      clearTyping();
      bubble.classList.add("error");
      bubble.textContent = message;
    },
  };
}

async function handleSubmit(ev) {
  ev.preventDefault();
  const message = el.input.value.trim();
  if (!message) return;

  el.input.value = "";
  el.sendBtn.disabled = true;
  renderMessage("user", message);

  const view = beginStreamingAssistantMessage();
  const splitter = makeThinkStreamSplitter(
    (text) => view.appendThinking(text),
    (text) => view.appendAnswer(text),
  );

  try {
    const resp = await api.streamMessage(state.sessionId, message);
    if (!resp.ok) {
      throw new Error(`stream ${resp.status}`);
    }

    let sources = [];
    for await (const evt of parseSSE(resp)) {
      if (evt.event === "session") {
        state.sessionId = evt.data.session_id;
        localStorage.setItem("session_id", state.sessionId);
      } else if (evt.event === "chunk") {
        splitter.feed(evt.data.text || "");
      } else if (evt.event === "done") {
        splitter.end();
        sources = evt.data.sources || [];
      } else if (evt.event === "error") {
        throw new Error(evt.data.message || "stream error");
      }
    }

    view.finalize(sources);
    renderSessionList();
  } catch (e) {
    splitter.end();
    view.fail(`Request failed: ${e.message}`);
  } finally {
    el.sendBtn.disabled = false;
    el.input.focus();
  }
}

async function refreshStatus() {
  try {
    const h = await api.health();
    el.status.textContent = `online · model: ${h.llm_model} · indexed chunks: ${h.vector_store_docs}`;
    el.status.className = "status online";
  } catch (e) {
    el.status.textContent = "backend offline";
    el.status.className = "status offline";
  }
}

// --------------------------------------------------------------------------- //
// Init                                                                        //
// --------------------------------------------------------------------------- //

async function init() {
  el.form.addEventListener("submit", handleSubmit);
  el.newChatBtn.addEventListener("click", startNewChat);

  await refreshStatus();
  await renderSessionList();

  if (state.sessionId) {
    await selectSession(state.sessionId);
  } else {
    renderEmptyState();
  }
}

init();
