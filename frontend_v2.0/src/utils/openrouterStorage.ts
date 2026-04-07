/** Browser persistence for OpenRouter API key (client-side only; never logged by app code). */

export const LS_OPENROUTER_KEY = 'rag_playground_openrouter_api_key';

export function loadStoredApiKey(): string {
  try {
    return localStorage.getItem(LS_OPENROUTER_KEY) ?? '';
  } catch {
    return '';
  }
}

export function saveStoredApiKey(key: string): void {
  try {
    if (!key.trim()) localStorage.removeItem(LS_OPENROUTER_KEY);
    else localStorage.setItem(LS_OPENROUTER_KEY, key.trim());
  } catch {
    /* quota / private mode */
  }
}

export function clearStoredApiKey(): void {
  try {
    localStorage.removeItem(LS_OPENROUTER_KEY);
  } catch {
    /* ignore */
  }
}
