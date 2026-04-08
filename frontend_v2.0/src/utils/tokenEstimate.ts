/** Rough OpenAI-style token estimate for teaching (not exact tokenizer). */
export const APPROX_CHARS_PER_TOKEN = 4;

export function approxTokensFromChars(charCount: number): number {
  if (!Number.isFinite(charCount) || charCount <= 0) return 0;
  return Math.max(1, Math.ceil(charCount / APPROX_CHARS_PER_TOKEN));
}

/** Typical instruction context for chat models (teaching default; not provider-specific). */
export const DEFAULT_TEACHING_CONTEXT_LIMIT = 128_000;
