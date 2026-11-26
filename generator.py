from typing import Any
from groq import Groq
from src.config import GROQ_MODEL_NAME, resolve_groq_api_key


class GroqRAGGenerator:
    """Generates answers from retrieved SOP + CV chunks using Groq."""

    def __init__(
        self,
        model_name: str = GROQ_MODEL_NAME,
        ui_api_key: str | None = None,
        secrets_obj: object | None = None,
    ) -> None:
        self.model_name = model_name
        api_key = resolve_groq_api_key(ui_key=ui_key, secrets_obj=secrets_obj)
        self.client = Groq(api_key=api_key)

    def _format_context(self, contexts: list[dict[str, Any]]) -> str:
        if not contexts:
            return "No relevant context was retrieved from the SOP or CV."

        parts: list[str] = []
        for idx, item in enumerate(contexts, start=1):
            meta = item.get("metadata", {})
            page = meta.get("page", "?")
            source = meta.get("source", "SOP+CV")
            parts.append(f"[Source {idx} | {source} | Page {page}]\n{item['text']}")
        return "\n\n---\n\n".join(parts)

    def generate(self, query: str, contexts: list[dict[str, Any]]) -> str:
        """Send retrieved SOP + CV chunks and user query to openai/gpt-oss-120b."""
        context_str = self._format_context(contexts)

        system_prompt = (
            "You are a Lead Generation & Outreach AI Assistant grounded strictly in "
            "Amir's Lead Generation & Outreach SOP + CV. Answer accurately using ONLY "
            "the provided context, citing page numbers like (Page 2) when referencing "
            "steps, tools, templates, or experience. If information is missing from "
            "the context, state that clearly."
        )

        user_prompt = (
            f"Retrieved Context:\n{context_str}\n\n"
            f"Query:\n{query}"
        )

        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.model_name,
        )

        content = chat_completion.choices[0].message.content
        return content.strip() if content else "No response returned by the model."
