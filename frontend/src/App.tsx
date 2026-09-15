import { FormEvent, useEffect, useRef, useState } from "react";
import { BookOpen, FileText, Search, ShieldCheck, Trash2, Upload } from "lucide-react";

type Document = {
  id: string;
  filename: string;
  pages: number;
  chunks: number;
  created_at: string;
};

type Citation = {
  number: number;
  source: string;
  page: number | null;
  text: string;
  score: number;
};

type Answer = {
  answer: string;
  grounded: boolean;
  mode: "ollama" | "extractive";
  citations: Citation[];
};

const API = import.meta.env.VITE_API_URL ?? "";
const EXAMPLE_QUESTIONS = [
  "What is the paper's main conclusion?",
  "Which limitations do the authors mention?",
  "What evidence supports the reported results?",
];

export default function App() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [modelReady, setModelReady] = useState<boolean | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  async function loadDocuments() {
    const response = await fetch(`${API}/api/documents`);
    if (response.ok) setDocuments(await response.json());
  }

  async function loadHealth() {
    const response = await fetch(`${API}/api/health`);
    if (response.ok) {
      const health = await response.json();
      setModelReady(health.model_ready);
    }
  }

  useEffect(() => {
    loadDocuments().catch(() => setMessage("Could not connect to the API."));
    loadHealth().catch(() => setModelReady(false));
  }, []);

  async function upload(file?: File) {
    if (!file) return;
    setBusy(true);
    setMessage("Extracting pages and creating embeddings…");
    // FormData sends the original binary PDF instead of trying to encode it as JSON.
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch(`${API}/api/documents`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Upload failed.");
      setMessage(`Indexed ${data.filename}: ${data.pages} pages, ${data.chunks} chunks.`);
      await loadDocuments();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setBusy(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function remove(id: string) {
    await fetch(`${API}/api/documents/${id}`, { method: "DELETE" });
    setAnswer(null);
    await loadDocuments();
  }

  async function ask(event: FormEvent) {
    event.preventDefault();
    if (!question.trim()) return;
    setBusy(true);
    setMessage("");
    try {
      // Retrieval is performed by the Python backend; the browser only renders
      // the ranked evidence and never receives or stores embedding vectors.
      const response = await fetch(`${API}/api/questions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 5 }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Question failed.");
      setAnswer(data);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Question failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <nav>
        <a className="brand" href="#"><BookOpen size={22} /> CiteWise</a>
        <span className={`badge ${modelReady === false ? "offline" : ""}`}>
          <ShieldCheck size={15} />
          {modelReady === null ? "Checking local model…" : modelReady ? "Local LLM ready" : "Ollama not ready"}
        </span>
      </nav>

      <section className="hero">
        <p className="eyebrow">PRIVATE · LOCAL · TRACEABLE</p>
        <h1>Research answers you can <em>verify.</em></h1>
        <p className="subtitle">Upload a PDF, ask a question, and inspect the exact passages behind every result.</p>
        <div className="built-by">A student-built RAG project by <strong>Luka Zotovikj</strong></div>
      </section>

      <section className="workspace">
        <aside className="panel documents">
          <div className="panel-title"><span>Your library</span><span>{documents.length}</span></div>
          <button className="upload" disabled={busy} onClick={() => fileInput.current?.click()}>
            <Upload size={18} /> Upload PDF
          </button>
          <input ref={fileInput} type="file" accept=".pdf,application/pdf" hidden onChange={(e) => upload(e.target.files?.[0])} />
          <div className="document-list">
            {documents.length === 0 && <p className="empty">Your uploaded research papers will appear here.</p>}
            {documents.map((doc) => (
              <article className="document" key={doc.id}>
                <FileText size={20} />
                <div><strong>{doc.filename}</strong><small>{doc.pages} pages · {doc.chunks} chunks</small></div>
                <button aria-label={`Delete ${doc.filename}`} onClick={() => remove(doc.id)}><Trash2 size={16} /></button>
              </article>
            ))}
          </div>
        </aside>

        <section className="panel search-panel">
          <form onSubmit={ask}>
            <label htmlFor="question">Ask your documents</label>
            <div className="search-box">
              <Search size={20} />
              <input id="question" value={question} onChange={(e) => setQuestion(e.target.value)}
                placeholder="What evidence supports the main conclusion?" />
              <button disabled={busy || !question.trim()}>Search</button>
            </div>
          </form>
          {message && <p className="message">{message}</p>}

          {!answer && !message && (
            <div className="welcome">
              <div className="orb"><Search size={32} /></div>
              <h2>Start with a question</h2>
              <p>CiteWise searches by meaning, not just matching keywords.</p>
              <div className="suggestions">
                {EXAMPLE_QUESTIONS.map((example) => (
                  <button key={example} onClick={() => setQuestion(example)}>{example}</button>
                ))}
              </div>
            </div>
          )}

          {answer && (
            <div className="answer">
              <div className="answer-heading">
                <span className={answer.grounded ? "grounded" : "ungrounded"}>
                  {answer.grounded ? "Evidence found" : "Insufficient evidence"}
                </span>
                <span className="mode">{answer.mode === "ollama" ? "Local LLM answer" : "Extractive mode"}</span>
              </div>
              <h2>{answer.answer}</h2>
              <div className="citations">
                {answer.citations.map((citation) => (
                  <article className="citation" key={citation.number}>
                    <header><span>[{citation.number}] {citation.source}</span><span>Page {citation.page ?? "—"}</span></header>
                    <p>{citation.text}</p>
                    <small>Hybrid relevance {(citation.score * 100).toFixed(1)}%</small>
                  </article>
                ))}
              </div>
            </div>
          )}
        </section>
      </section>
      <section className="how-it-works">
        <article><span>01</span><strong>Extract</strong><p>Read text while preserving PDF page numbers.</p></article>
        <article><span>02</span><strong>Embed</strong><p>Represent questions and passages as semantic vectors.</p></article>
        <article><span>03</span><strong>Retrieve</strong><p>Rank evidence and return traceable citations.</p></article>
      </section>
      <footer>Designed and built by Luka Zotovikj · FastAPI · React · Sentence Transformers</footer>
    </main>
  );
}
