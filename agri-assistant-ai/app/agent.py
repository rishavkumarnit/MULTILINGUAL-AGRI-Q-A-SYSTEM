"""Tool-calling agent: the model decides which tools to call, if any, instead of a
hand-coded intent classification. Grounding is enforced in code, not just prompted —
if no tool call returns usable data, the caller must fall back to a non-generated
status message rather than trust the model's own text."""

import json
import os
from dataclasses import dataclass

from openai import AsyncOpenAI

from .crop_price import fetch_crop_price, format_price
from .rag import RetrievedChunk, find_relevant_chunks
from .weather import fetch_weather_forecast, format_forecast

MAX_TOOL_ROUNDS = 4
ANSWER_TOKEN_CAP = 300  # forces brevity; the model otherwise ignores length instructions

AGENT_INSTRUCTIONS = (
    "You are an agricultural assistant for Indian farmers. You have tools for weather "
    "forecasts, crop market prices, and searching trusted agricultural documents. Call "
    "whichever tools would help answer the question — none, one, or several. "
    "Only state facts that came from a tool result. If no tool call returns useful "
    "information, say plainly that you don't have grounded information to answer, "
    "rather than guessing from general knowledge. "
    "Keep the answer very short: at most 5-10 sentences total. If the question needs "
    "step-by-step guidance, use at most one bulleted list of up to 5 items, one short "
    "sentence each, instead of the sentences — never both. Never use headings, numbered "
    "sections, or multiple lists; give the single most useful answer directly instead of "
    "a full guide covering every angle. Do not restate the question, add disclaimers, "
    "or ask a follow-up question unless essential information is missing. Return only "
    "the answer in plain English."
)

TOOLS = [
    {
        "type": "function",
        "name": "get_weather_forecast",
        "description": "Get a 5-day weather forecast for a city, district, or place name.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_crop_price",
        "description": "Get the latest mandi (market) price for a crop, optionally scoped to a district or state.",
        "parameters": {
            "type": "object",
            "properties": {
                "crop": {"type": "string"},
                "location": {"type": ["string", "null"], "description": "District or state, or null if not known."},
            },
            "required": ["crop", "location"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_documents",
        "description": "Search trusted agricultural documents (irrigation, pest management, soil health, etc.) for information relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


@dataclass
class AgentResult:
    answer_english: str
    source: str
    sources: list[str] | None = None


async def run_agent(question_english: str, crop: str | None, location: str | None) -> AgentResult:
    """Let the model decide which tools to call, then synthesize an answer.

    Falls back to source="ungrounded" (never exposed on ChatResponse) if no tool call
    returned usable data, regardless of what text the model produced.
    """
    client = AsyncOpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    context_lines = [f"Question: {question_english}"]
    if crop:
        context_lines.append(f"Crop mentioned: {crop}")
    if location:
        context_lines.append(f"Location mentioned: {location}")

    forecast_found = None
    price_found = None
    chunks_found: list[RetrievedChunk] = []

    response = await client.responses.create(
        model=model, instructions=AGENT_INSTRUCTIONS, input="\n".join(context_lines), tools=TOOLS,
        reasoning={"effort": "minimal"}, max_output_tokens=ANSWER_TOKEN_CAP,
    )

    for _ in range(MAX_TOOL_ROUNDS):
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            break
        outputs = []
        for call in calls:
            arguments = json.loads(call.arguments)
            result_text, forecast, price, chunks = await _execute_tool(call.name, arguments)
            forecast_found = forecast_found or forecast
            price_found = price_found or price
            chunks_found.extend(chunks)
            outputs.append({"type": "function_call_output", "call_id": call.call_id, "output": result_text})
        response = await client.responses.create(
            model=model, previous_response_id=response.id, input=outputs, tools=TOOLS,
            reasoning={"effort": "minimal"}, max_output_tokens=ANSWER_TOKEN_CAP,
        )

    answer_text = _trim_if_cut_off(response.output_text, getattr(response, "status", None))

    if forecast_found:
        return AgentResult(answer_english=answer_text or format_forecast(forecast_found), source="weather-forecast")
    if price_found:
        return AgentResult(answer_english=answer_text or format_price(price_found), source="price-lookup")
    if chunks_found:
        return AgentResult(
            answer_english=answer_text or "\n\n".join(chunk.text for chunk in chunks_found),
            source="rag-generated",
            sources=sorted({chunk.title for chunk in chunks_found}),
        )
    return AgentResult(answer_english="", source="ungrounded")


def _trim_if_cut_off(text: str, status: str | None) -> str:
    """If generation hit the token cap mid-sentence, trim back to the last complete
    sentence rather than showing a sentence broken off mid-word."""
    if status != "incomplete" or not text:
        return text
    last_boundary = max(text.rfind(". "), text.rfind("? "), text.rfind("! "), text.rfind("\n"))
    return text[: last_boundary + 1].rstrip() if last_boundary > 0 else text


async def _execute_tool(name: str, arguments: dict):
    """Returns (result_text_for_model, forecast_or_none, price_or_none, chunks)."""
    if name == "get_weather_forecast":
        forecast = await fetch_weather_forecast(arguments["location"])
        if not forecast:
            return f"No weather data found for '{arguments['location']}'.", None, None, []
        return format_forecast(forecast), forecast, None, []

    if name == "get_crop_price":
        price = await fetch_crop_price(arguments["crop"], arguments.get("location") or None)
        if not price:
            where = f" in '{arguments['location']}'" if arguments.get("location") else ""
            return f"No price data found for '{arguments['crop']}'{where}.", None, None, []
        return format_price(price), None, price, []

    if name == "search_documents":
        chunks = await find_relevant_chunks(arguments["query"])
        if not chunks:
            return "No relevant documents found.", None, None, []
        text = "\n\n".join(f"[{chunk.title}]\n{chunk.text}" for chunk in chunks)
        return text, None, None, chunks

    return f"Unknown tool: {name}", None, None, []
