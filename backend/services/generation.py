"""LLM generation service — template-based for Phase 1, pluggable interface."""
from typing import Optional
from openai import OpenAI


class BaseGenerator:
    """Abstract generator interface for pluggable LLM backends."""

    def generate(self, query: str, context_chunks: list[str], **kwargs) -> tuple[str, dict]:
        raise NotImplementedError


class TemplateGenerator(BaseGenerator):
    """Template-based generation for Phase 1 (no LLM API required)."""

    def generate(
        self,
        query: str,
        context_chunks: list[str],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> tuple[str, dict]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        assembled = kwargs.get("assembled_prompt")
        if assembled:
            answer = (
                f"[Template mode] Prompt length {len(assembled)} chars.\n\n"
                f"---\n{assembled[:2000]}{'…' if len(assembled) > 2000 else ''}\n---\n"
                f"*Connect an LLM provider for full generation.*"
            )
            return answer, usage

        if not context_chunks:
            return "No relevant documents found. Please upload documents first.", usage

        # Build a structured answer from retrieved context
        context_summary = self._build_context_summary(context_chunks)

        answer = (
            f"Based on the retrieved context, here is the information relevant to your query:\n\n"
            f"**Query:** {query}\n\n"
            f"**Retrieved Context Summary:**\n\n{context_summary}\n\n"
            f"---\n"
            f"*This answer was generated using template-based synthesis (Phase 1). "
            f"Connect an LLM provider for AI-generated answers.*"
        )

        return answer, usage

    def _build_context_summary(self, chunks: list[str]) -> str:
        """Build a readable summary from context chunks."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            # Truncate very long chunks for readability
            preview = chunk[:500] + "..." if len(chunk) > 500 else chunk
            parts.append(f"**[Source {i}]** {preview}")
        return "\n\n".join(parts)


class OpenRouterGenerator(BaseGenerator):
    """Actual LLM generator using OpenRouter for Phase 2."""

    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def generate(
        self,
        query: str,
        context_chunks: list[str],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> tuple[str, dict]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        assembled = kwargs.get("assembled_prompt")
        if assembled:
            system_prompt = (
                "You are a helpful AI assistant. Answer using the user message, which may include "
                "retrieved context and a question."
            )
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": str(assembled)},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    content = "Error: No response content from OpenRouter."
                if response.usage:
                    usage["prompt_tokens"] = response.usage.prompt_tokens or 0
                    usage["completion_tokens"] = response.usage.completion_tokens or 0
                    usage["total_tokens"] = response.usage.total_tokens or 0
                return content, usage
            except Exception as e:
                return f"Error generating response via OpenRouter: {str(e)}", usage

        if not context_chunks:
            return "No relevant documents found. Please upload documents first.", usage

        context_text = "\n\n".join([f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])

        system_prompt = (
            "You are a helpful AI assistant. Use the following retrieved context to answer the user's query. "
            "If the answer cannot be deduced from the context, state that you do not have enough information."
        )

        user_prompt = f"Context:\n{context_text}\n\nQuery:\n{query}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                content = "Error: No response content from OpenRouter."
                
            if response.usage:
                usage["prompt_tokens"] = response.usage.prompt_tokens
                usage["completion_tokens"] = response.usage.completion_tokens
                usage["total_tokens"] = response.usage.total_tokens
                
            return content, usage
        except Exception as e:
            return f"Error generating response via OpenRouter: {str(e)}", usage


# Global generator instance
_generator: Optional[BaseGenerator] = None


def get_generator(api_key: Optional[str] = None, model: str = "") -> BaseGenerator:
    """Get the current generator instance."""
    global _generator
    
    if api_key and api_key.strip():
        # Instantiate OpenRouterGenerator if missing or config changed
        if not isinstance(_generator, OpenRouterGenerator) or _generator.model != model or _generator.client.api_key != api_key:
            _generator = OpenRouterGenerator(api_key=api_key.strip(), model=model)
    else:
        # Fallback to TemplateGenerator
        if not isinstance(_generator, TemplateGenerator):
            _generator = TemplateGenerator()
            
    return _generator


def generate_answer(
    query: str,
    context_chunks: list[str],
    temperature: float = 0.7,
    max_tokens: int = 512,
    openrouter_api_key: Optional[str] = None,
    llm_model: str = "openai/gpt-4o-mini",
    assembled_prompt: Optional[str] = None,
) -> tuple[str, dict]:
    """Generate an answer given query and context chunks, or a single assembled prompt from Prompt Augment."""
    gen = get_generator(api_key=openrouter_api_key, model=llm_model)
    return gen.generate(
        query=query,
        context_chunks=context_chunks,
        temperature=temperature,
        max_tokens=max_tokens,
        assembled_prompt=assembled_prompt,
    )
