import { FormEvent, useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
  context?: { crop?: string | null; location?: string | null; similarity?: number | null; sources?: string[] | null; source?: string | null };
};

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
  const [conversationId, setConversationId] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const value = question.trim();
    if (!value || isLoading) return;

    setMessages((current) => [...current, { role: "user", content: value }]);
    setQuestion("");
    setError("");
    setIsLoading(true);

    try {
      const result = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: value, language, conversationId })
      });

      if (!result.ok || !result.body) {
        let message = `The API returned an unexpected response (HTTP ${result.status}). Ensure the AI service is running on port 8000.`;
        try {
          const body = JSON.parse(await result.text()) as { error?: string };
          if (body.error) message = body.error;
        } catch {
          // keep the default message above
        }
        throw new Error(message);
      }

      let assistantAdded = false;
      let buffer = "";
      const reader = result.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value: chunk } = await reader.read();
        if (done) break;
        buffer += decoder.decode(chunk, { stream: true });

        let separatorIndex: number;
        while ((separatorIndex = buffer.indexOf("\n\n")) !== -1) {
          const rawEvent = buffer.slice(0, separatorIndex);
          buffer = buffer.slice(separatorIndex + 2);
          const line = rawEvent.split("\n").find((entry) => entry.startsWith("data: "));
          if (!line) continue;
          const streamEvent = JSON.parse(line.slice("data: ".length)) as {
            type: string; text?: string; conversationId?: string;
            crop?: string | null; location?: string | null; similarity?: number | null; sources?: string[] | null; source?: string | null;
          };

          if (streamEvent.type === "metadata") {
            assistantAdded = true;
            if (streamEvent.conversationId) setConversationId(streamEvent.conversationId);
            setMessages((current) => [...current, {
              role: "assistant", content: "",
              context: { crop: streamEvent.crop, location: streamEvent.location, similarity: streamEvent.similarity, sources: streamEvent.sources, source: streamEvent.source }
            }]);
          } else if (streamEvent.type === "delta") {
            setMessages((current) => {
              const next = [...current];
              const last = next[next.length - 1];
              if (last?.role === "assistant") next[next.length - 1] = { ...last, content: last.content + (streamEvent.text ?? "") };
              return next;
            });
          }
        }
      }

      if (!assistantAdded) throw new Error("No response received from the assistant.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to get an answer.");
    } finally {
      setIsLoading(false);
    }
  }

  const isStreamingAssistantMessage = isLoading && messages[messages.length - 1]?.role === "assistant";

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

    <p className="disclaimer">
      This is a personal learning/portfolio project, not a real agricultural advisory service.
      Verify important decisions with your local agricultural extension office.
    </p>

    <section className="chat" aria-live="polite">
      {messages.length === 0 && <div className="empty-state">
        <span className="seedling">🌱</span>
        <h2>What can I help you grow?</h2>
        <p>Ask about crops, pests, irrigation, soil, or planting.</p>
      </div>}
      {messages.map((message, index) => <article key={index} className={`message ${message.role}`}>
        <span>{message.role === "user" ? "You" : "Krishi Sahayak"}</span>
        <p>{message.content}</p>
        {message.role === "assistant" && (message.context?.crop || message.context?.location || message.context?.similarity || message.context?.sources?.length || message.context?.source === "llm-general") && <div className="context">
          {message.context.source === "llm-general" && <span className="badge-warning">General AI answer — not verified against our database</span>}
          {message.context.crop && <span>Crop: {message.context.crop}</span>}
          {message.context.location && <span>Location: {message.context.location}</span>}
          {message.context.similarity && <span>Verified match: {Math.round(message.context.similarity * 100)}%</span>}
          {message.context.sources && message.context.sources.length > 0 && <span>Sources: {message.context.sources.join(", ")}</span>}
        </div>}
      </article>)}
      {isLoading && !isStreamingAssistantMessage && <article className="message assistant"><span>Krishi Sahayak</span><p>Thinking…</p></article>}
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
