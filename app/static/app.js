// ECG Knowledge Chatbot — client-side chat UI.
// History is in-memory and per-session; it is sent with every /chat call.

const state = {
  history: [],
  citations: [],
  busy: false,
};

const messagesEl = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const clearBtn = document.getElementById("clear");
const modal = document.getElementById("preview-modal");
const previewTitle = document.getElementById("preview-title");
const previewBody = document.getElementById("preview-body");
const previewClose = document.getElementById("preview-close");

function textNode(text) {
  return document.createTextNode(text);
}

function citationChip(idx) {
  const citation = state.citations[idx];
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "cite";
  btn.textContent = "[" + (idx + 1) + "]";
  if (!citation) return btn;
  btn.title =
    citation.heading_path ||
    citation.path + (citation.section ? " \u00a7" + citation.section : "");
  btn.addEventListener("click", () => openCitation(citation));
  return btn;
}

// Render the assistant answer, turning inline [n] markers into citation chips.
function renderAnswer(text) {
  const container = document.createElement("div");
  container.className = "answer";
  const parts = text.split(/(\[\d+\])/g);
  for (const part of parts) {
    const match = /^\[(\d+)\]$/.exec(part);
    if (match) {
      container.appendChild(citationChip(parseInt(match[1], 10) - 1));
    } else if (part) {
      container.appendChild(textNode(part));
    }
  }
  return container;
}

function addMessage(role, opts = {}) {
  const el = document.createElement("div");
  el.className = "message " + role;
  if (opts.pending) {
    el.classList.add("pending");
    el.textContent = "Thinking";
  } else if (role === "user" || opts.error) {
    el.textContent = opts.text;
  } else {
    el.appendChild(renderAnswer(opts.text));
  }
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

async function openCitation(citation) {
  if (citation.url) {
    window.open(citation.url, "_blank", "noopener");
    return;
  }
  try {
    const res = await fetch("/doc/" + encodeURI(citation.path));
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    showPreview(citation.path, data.markdown);
  } catch (err) {
    showPreview(citation.path, "Could not load document preview: " + err.message);
  }
}

function showPreview(title, body) {
  previewTitle.textContent = title;
  previewBody.textContent = body;
  modal.classList.add("open");
}

function closePreview() {
  modal.classList.remove("open");
}

async function send(text) {
  if (state.busy) return;
  text = (text || "").trim();
  if (!text) return;
  input.value = "";
  autoGrow();
  addMessage("user", { text });
  state.busy = true;
  sendBtn.disabled = true;

  const pendingEl = addMessage("assistant", { pending: true });
  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: text,
        history: state.history.slice(-10),
        k: 8,
      }),
    });
    let payload = null;
    try {
      payload = await resp.json();
    } catch (_) {
      // non-JSON error body
    }
    if (!resp.ok) {
      throw new Error(
        payload && payload.detail ? payload.detail : "HTTP " + resp.status
      );
    }
    state.citations = payload.citations || [];
    pendingEl.classList.remove("pending");
    pendingEl.replaceChildren(renderAnswer(payload.answer || ""));
    state.history.push(
      { role: "user", content: text },
      { role: "assistant", content: payload.answer || "" }
    );
  } catch (err) {
    pendingEl.classList.remove("pending");
    pendingEl.classList.add("error");
    pendingEl.textContent = "Error: " + err.message;
    state.history.push({ role: "user", content: text });
  } finally {
    state.busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  send(input.value);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", autoGrow);

clearBtn.addEventListener("click", () => {
  state.history = [];
  state.citations = [];
  messagesEl.replaceChildren();
  input.focus();
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    if (!state.busy) send(chip.dataset.q);
  });
});

previewClose.addEventListener("click", closePreview);
modal.addEventListener("click", (e) => {
  if (e.target === modal) closePreview();
});

addMessage("assistant", {
  text: "Hello! Ask me anything about the ECG analysis pipeline \u2014 signal quality, feature extraction, or the classification models.",
});
