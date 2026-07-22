import cors from "cors";
import "dotenv/config";
import express, { type Request, type Response } from "express";
import { randomUUID } from "node:crypto";
import { MongoClient, ServerApiVersion, type Db } from "mongodb";

const app = express();
const port = Number(process.env.PORT ?? 4000);
const aiServiceUrl = process.env.AI_SERVICE_URL ?? "http://localhost:8000";
let database: Db | null = null;

app.use(cors());
app.use(express.json({ limit: "100kb" }));

type ChatRequest = {
  question?: string;
  language?: string;
  conversationId?: string;
};

const languageNames: Record<string, string> = {
  en: "English",
  hi: "Hindi",
  bn: "Bengali",
  ta: "Tamil",
  te: "Telugu",
  mr: "Marathi"
};

app.get("/api/health", (_request: Request, response: Response) => {
  response.json({ status: "ok", database: database ? "connected" : "not-configured" });
});

app.post("/api/chat", async (request: Request<{}, {}, ChatRequest>, response: Response) => {
  const question = request.body.question?.trim();
  const language = languageNames[request.body.language ?? "en"] ? request.body.language ?? "en" : "en";

  if (!question) {
    response.status(400).json({ error: "A question is required." });
    return;
  }

  try {
    const conversationId = request.body.conversationId ?? randomUUID();
    await persistMessage(conversationId, {
      role: "user",
      content: question,
      language,
      createdAt: new Date()
    });
    const aiResponse = await fetch(`${aiServiceUrl}/internal/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, language, conversationId })
    });
    const payload = await aiResponse.json() as {
      answer?: string; source?: string; questionEnglish?: string; crop?: string | null; location?: string | null; similarity?: number | null; sources?: string[] | null; detail?: string
    };
    if (!aiResponse.ok) throw new Error(payload.detail ?? "AI service request failed");

    await persistMessage(conversationId, {
      role: "assistant",
      content: payload.answer ?? "",
      questionEnglish: payload.questionEnglish,
      crop: payload.crop,
      location: payload.location,
      source: payload.source,
      similarity: payload.similarity,
      sources: payload.sources,
      createdAt: new Date()
    });
    response.json({
      conversationId,
      answer: payload.answer,
      language,
      source: payload.source,
      questionEnglish: payload.questionEnglish,
      crop: payload.crop,
      location: payload.location,
      similarity: payload.similarity,
      sources: payload.sources
    });
  } catch (error) {
    console.error("AI service request failed", error);
    response.status(503).json({ error: "The AI service is unavailable. Start the FastAPI service and try again." });
  }
});

app.post("/api/chat/stream", async (request: Request<{}, {}, ChatRequest>, response: Response) => {
  const question = request.body.question?.trim();
  const language = languageNames[request.body.language ?? "en"] ? request.body.language ?? "en" : "en";

  if (!question) {
    response.status(400).json({ error: "A question is required." });
    return;
  }

  const conversationId = request.body.conversationId ?? randomUUID();

  try {
    await persistMessage(conversationId, {
      role: "user",
      content: question,
      language,
      createdAt: new Date()
    });

    const aiResponse = await fetch(`${aiServiceUrl}/internal/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, language, conversationId })
    });
    if (!aiResponse.ok || !aiResponse.body) throw new Error("AI service stream request failed");

    response.setHeader("Content-Type", "text/event-stream");
    response.setHeader("Cache-Control", "no-cache");
    response.setHeader("Connection", "keep-alive");
    response.flushHeaders();

    let answerText = "";
    let metadata: Record<string, unknown> = {};
    let buffer = "";
    const reader = aiResponse.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let separatorIndex: number;
      while ((separatorIndex = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const line = rawEvent.split("\n").find((entry) => entry.startsWith("data: "));
        if (!line) continue;
        const event = JSON.parse(line.slice("data: ".length)) as { type: string; text?: string; [key: string]: unknown };

        if (event.type === "metadata") {
          metadata = event;
          response.write(`data: ${JSON.stringify({ ...event, conversationId })}\n\n`);
        } else {
          if (event.type === "delta") answerText += event.text ?? "";
          response.write(`data: ${JSON.stringify(event)}\n\n`);
        }
      }
    }

    await persistMessage(conversationId, {
      role: "assistant",
      content: answerText,
      questionEnglish: metadata.questionEnglish,
      crop: metadata.crop,
      location: metadata.location,
      source: metadata.source,
      similarity: metadata.similarity,
      sources: metadata.sources,
      createdAt: new Date()
    });
    response.end();
  } catch (error) {
    console.error("AI service stream request failed", error);
    if (!response.headersSent) {
      response.status(503).json({ error: "The AI service is unavailable. Start the FastAPI service and try again." });
    } else {
      response.end();
    }
  }
});

async function persistMessage(conversationId: string, message: Record<string, unknown>) {
  if (!database) return;
  const now = new Date();
  await database.collection("conversations").updateOne(
    { _id: conversationId },
    {
      $setOnInsert: { createdAt: now },
      $set: { updatedAt: now },
      $push: { messages: message }
    },
    { upsert: true }
  );
}

async function start() {
  const uri = process.env.MONGODB_URI;
  if (uri) {
    const client = new MongoClient(uri, {
      serverApi: { version: ServerApiVersion.v1, strict: true, deprecationErrors: true }
    });
    await client.connect();
    database = client.db(process.env.MONGODB_DATABASE ?? "agri_assistant");
    console.log("MongoDB Atlas connected");
  } else {
    console.warn("MONGODB_URI is not configured; conversation persistence is disabled.");
  }

  app.listen(port, () => {
    console.log(`Agri Assistant API listening on http://localhost:${port}`);
  });
}

start().catch((error) => {
  console.error("API startup failed", error);
  process.exit(1);
});
