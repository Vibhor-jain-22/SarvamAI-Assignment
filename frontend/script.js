const API_BASE = "http://127.0.0.1:8000";

const chatEl = document.getElementById("chat");
const statusEl = document.getElementById("status");
const pdfInput = document.getElementById("pdfInput");
const imageInput = document.getElementById("imageInput");
const questionInput = document.getElementById("question");
const sendBtn = document.getElementById("sendBtn");

let manualReady = false;

function addMessage(role, content, sources) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role === "user" ? "user" : "bot"}`;

  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role === "user" ? "You" : "Assistant";

  const contentEl = document.createElement("div");
  contentEl.className = "content";
  contentEl.textContent = content || "";

  wrapper.appendChild(roleEl);
  wrapper.appendChild(contentEl);

  if (role !== "user" && Array.isArray(sources)) {
    const sourcesEl = document.createElement("div");
    sourcesEl.className = "sources";

    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = sources.length ? `Sources (${sources.length})` : "Sources (none)";
    details.appendChild(summary);

    sources.forEach((s, idx) => {
      const item = document.createElement("div");
      item.className = "sourceItem";
      const meta = document.createElement("div");
      meta.className = "sourceMeta";
      meta.textContent = `Page ${s.page}`;
      const txt = document.createElement("div");
      txt.className = "content";
      txt.textContent = s.text || "";
      item.appendChild(meta);
      item.appendChild(txt);
      details.appendChild(item);
    });

    sourcesEl.appendChild(details);
    wrapper.appendChild(sourcesEl);
  }

  chatEl.appendChild(wrapper);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function setStatus(text, ok) {
  statusEl.textContent = text;
  statusEl.style.color = ok ? "var(--accent2)" : "var(--muted)";
}

async function uploadPdf(file) {
  manualReady = false;
  sendBtn.disabled = true;

  const fd = new FormData();
  fd.append("pdf", file);

  const data = await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/upload`);

    xhr.upload.onprogress = (evt) => {
      if (!evt.lengthComputable) {
        setStatus("Uploading manual…", false);
        return;
      }
      const pct = Math.max(0, Math.min(100, Math.round((evt.loaded / evt.total) * 100)));
      setStatus(`Uploading manual… ${pct}%`, false);
    };

    xhr.onload = () => {
      let payload = {};
      try {
        payload = JSON.parse(xhr.responseText || "{}");
      } catch (_) {
        payload = {};
      }
      if (xhr.status >= 200 && xhr.status < 300) return resolve(payload);
      return reject(new Error(payload.detail || "Upload failed"));
    };
    xhr.onerror = () => reject(new Error("Network error during upload"));

    // Upload is done; backend now indexes synchronously before responding.
    xhr.upload.onloadend = () => {
      setStatus("Indexing manual…", false);
    };

    xhr.send(fd);
  });

  manualReady = true;
  setStatus(`Manual indexed: ${data.manifest?.last_uploaded_filename || "uploaded.pdf"}`, true);
  sendBtn.disabled = false;
}

async function queryText(question) {
  const fd = new FormData();
  fd.append("question", question);
  const res = await fetch(`${API_BASE}/query`, { method: "POST", body: fd });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || "Query failed");
  return data;
}

async function queryImage(file) {
  const fd = new FormData();
  fd.append("image", file);
  const res = await fetch(`${API_BASE}/query`, { method: "POST", body: fd });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || "Image query failed");
  return data;
}

pdfInput.addEventListener("change", async (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  try {
    await uploadPdf(f);
    addMessage("bot", "Manual uploaded and indexed. Ask a question from the manual.", []);
  } catch (err) {
    setStatus("Upload failed", false);
    addMessage("bot", `Upload error: ${err.message}`, []);
  } finally {
    pdfInput.value = "";
  }
});

imageInput.addEventListener("change", async (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  if (!manualReady) {
    addMessage("bot", "Please upload a manual PDF first.", []);
    imageInput.value = "";
    return;
  }

  addMessage("user", "[Image uploaded]", null);
  try {
    const data = await queryImage(f);
    addMessage("bot", `${data.answer || ""}`, data.sources || []);
  } catch (err) {
    addMessage("bot", `Error: ${err.message}`, []);
  } finally {
    imageInput.value = "";
  }
});

sendBtn.addEventListener("click", async () => {
  const q = (questionInput.value || "").trim();
  if (!q) return;
  addMessage("user", q, null);
  questionInput.value = "";

  try {
    const data = await queryText(q);
    addMessage("bot", data.answer || "", data.sources || []);
  } catch (err) {
    addMessage("bot", `Error: ${err.message}`, []);
  }
});

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendBtn.click();
});

sendBtn.disabled = true;
setStatus("No manual uploaded", false);
addMessage("bot", "Upload a bike manual PDF to begin.", []);

