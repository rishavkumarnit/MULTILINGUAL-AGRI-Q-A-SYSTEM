"""Token-accurate chunking for RAG document ingestion."""

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_tokens: int = 300, overlap_tokens: int = 50) -> list[str]:
    """Split text into overlapping, token-bounded chunks.

    Paragraphs are packed together up to chunk_tokens first; a single
    paragraph longer than chunk_tokens is sliced with a sliding token
    window so no chunk exceeds the target size.
    """
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]

    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_tokens = 0

    def flush() -> None:
        if current_paragraphs:
            chunks.append("\n\n".join(current_paragraphs))

    for paragraph in paragraphs:
        paragraph_tokens = len(_ENCODING.encode(paragraph))
        if paragraph_tokens > chunk_tokens:
            flush()
            current_paragraphs, current_tokens = [], 0
            chunks.extend(_slide_window(paragraph, chunk_tokens, overlap_tokens))
            continue
        if current_tokens + paragraph_tokens > chunk_tokens:
            flush()
            current_paragraphs, current_tokens = [], 0
        current_paragraphs.append(paragraph)
        current_tokens += paragraph_tokens

    flush()
    return chunks


def _slide_window(paragraph: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    tokens = _ENCODING.encode(paragraph)
    step = max(chunk_tokens - overlap_tokens, 1)
    windows = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_tokens]
        windows.append(_ENCODING.decode(window))
        if start + chunk_tokens >= len(tokens):
            break
    return windows
