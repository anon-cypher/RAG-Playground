/** Mirrors backend prompt_augment assembly for live inspector preview. */

export function assemblePrompt(
  config: Record<string, unknown>,
  query: string,
  contextTexts: string[],
): string {
  const sep = String(config['context_separator'] ?? '\n\n---\n\n');
  const systemP = String(
    config['system_prompt'] ?? 'You are a helpful assistant. Use the context to answer.',
  );
  const userT = String(
    config['user_template'] ?? 'Context:\n{context}\n\nQuestion:\n{query}',
  );
  const numbered = contextTexts.map((t, i) => `[${i + 1}] ${t}`);
  const ctxBlock = numbered.length
    ? numbered.join(sep)
    : '(no retrieved context yet)';
  const userFilled = userT.replace(/\{query\}/g, query).replace(/\{context\}/g, ctxBlock);
  return `${systemP}\n\n${userFilled}`.trim();
}
