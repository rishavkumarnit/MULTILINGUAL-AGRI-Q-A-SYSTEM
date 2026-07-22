import { FormEvent, useState } from "react";

type Message = { role: "user" | "assistant"; content: string; context?: { crop?: string | null; location?: string | null; similarity?: number | null; sources?: string[] | null } };

const languages = [
  ["en", "English"], ["hi", "हिन्दी"], ["bn", "বাংলা"],
  ["ta", "தமிழ்"], ["te", "తెలుగు"], ["mr", "मराठी"]
] as const;

export default function App() {
  const [language, setLanguage] = useState("en");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    const value = question.trim();
    if (!value || isLoading) return;

    setMessages((current) => [...current, { role: "user", content: value }]);
    setQuestion("");
    setError("");
    setIsLoading(true);
    try {
      const result = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: value, language })
      });
      const responseText = await result.text();
      let body: { answer?: string; error?: string; crop?: string | null; location?: string | null; similarity?: number | null; sources?: string[] | null };
      try {
        body = JSON.parse(responseText) as typeof body;
      } catch {
        throw new Error(`The API returned an unexpected response (HTTP ${result.status}). Ensure the correct Node backend is running on port 4000.`);
      }
      if (!result.ok) throw new Error(body.error ?? "Unable to get an answer.");
      setMessages((current) => [...current, {
        role: "assistant", content: body.answer ?? "",
        context: { crop: body.crop, location: body.location, similarity: body.similarity, sources: body.sources }
      }]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to get an answer.");
    } finally {
      setIsLoading(false);
    }
  }

  return <main className="page">
    <section className="hero">
      <span className="eyebrow">MULTILINGUAL AGRICULTURAL ASSISTANT</span>
      <p>Practical farming guidance, in the language you choose.</p>
      <label className="language-label">Your language
        <select value={language} onChange={(event) => setLanguage(event.target.value)}>
          {languages.map(([code, label]) => <option key={code} value={code}>{label}</option>)}
        </select>
      </label>
    </section>

    <section className="chat" aria-live="polite">
      {messages.length === 0 && <div className="empty-state">
        <span className="seedling">🌱</span>
        <h2>What can I help you grow?</h2>
        <p>Ask about crops, pests, irrigation, soil, or planting.</p>
      </div>}
      {messages.map((message, index) => <article key={index} className={`message ${message.role}`}>
        <span>{message.role === "user" ? "You" : "Krishi Sahayak"}</span>
        <p>{message.content}</p>
        {message.role === "assistant" && (message.context?.crop || message.context?.location || message.context?.similarity || message.context?.sources?.length) && <div className="context">
          {message.context.crop && <span>Crop: {message.context.crop}</span>}
          {message.context.location && <span>Location: {message.context.location}</span>}
          {message.context.similarity && <span>Verified match: {Math.round(message.context.similarity * 100)}%</span>}
          {message.context.sources && message.context.sources.length > 0 && <span>Sources: {message.context.sources.join(", ")}</span>}
        </div>}
      </article>)}
      {isLoading && <article className="message assistant"><span>Krishi Sahayak</span><p>Thinking…</p></article>}
    </section>

    <form className="composer" onSubmit={submit}>
      <label htmlFor="question">Ask a farming question</label>
      <div className="input-row">
        <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="e.g. How should I manage aphids on mustard?" rows={2} />
        <button type="submit" disabled={isLoading || !question.trim()}>{isLoading ? "Sending" : "Ask"}</button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
    </form>
  </main>;
}
