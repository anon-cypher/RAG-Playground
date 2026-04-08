import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import {
  clearStoredApiKey,
  loadStoredApiKey,
  saveStoredApiKey,
} from '../utils/openrouterStorage';

type SettingsBarProps = {
  onCredentialsSynced?: () => void;
};

/**
 * OpenRouter API key: persisted in localStorage and synced to the backend (in-memory) without rebuilding the index.
 */
export function SettingsBar({ onCredentialsSynced }: SettingsBarProps) {
  const [keyInput, setKeyInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const restoredOnce = useRef(false);

  useEffect(() => {
    const stored = loadStoredApiKey();
    setKeyInput(stored);
    if (!stored.trim() || restoredOnce.current) return;
    restoredOnce.current = true;
    setBusy(true);
    api
      .setCredentials(stored)
      .then(() => {
        setMsg('Saved API key restored to server.');
        onCredentialsSynced?.();
      })
      .catch((e: Error) => setErr(e.message))
      .finally(() => setBusy(false));
  }, [onCredentialsSynced]);

  const save = useCallback(() => {
    setErr(null);
    setMsg(null);
    setBusy(true);
    saveStoredApiKey(keyInput);
    api
      .setCredentials(keyInput.trim() || null)
      .then(() => {
        setMsg('API key saved (local + server).');
        onCredentialsSynced?.();
      })
      .catch((e: Error) => setErr(e.message))
      .finally(() => setBusy(false));
  }, [keyInput, onCredentialsSynced]);

  const testKey = useCallback(() => {
    setErr(null);
    setMsg(null);
    if (!keyInput.trim()) {
      setErr('Enter a key to test.');
      return;
    }
    setBusy(true);
    api
      .verifyOpenRouterKey(keyInput.trim())
      .then(() => setMsg('OpenRouter accepted this key.'))
      .catch((e: Error) => setErr(e.message))
      .finally(() => setBusy(false));
  }, [keyInput]);

  const clearKey = useCallback(() => {
    setErr(null);
    setMsg(null);
    clearStoredApiKey();
    setKeyInput('');
    setBusy(true);
    api
      .setCredentials(null)
      .then(() => {
        setMsg('API key cleared from this browser and server session.');
        onCredentialsSynced?.();
      })
      .catch((e: Error) => setErr(e.message))
      .finally(() => setBusy(false));
  }, [onCredentialsSynced]);

  return (
    <div className="settings-bar" role="region" aria-label="OpenRouter settings">
      <div className="settings-bar__row">
        <label className="settings-bar__label" htmlFor="openrouter-key-input">
          OpenRouter API key
        </label>
        <input
          id="openrouter-key-input"
          className="settings-bar__input"
          type="password"
          autoComplete="off"
          placeholder="sk-or-…"
          value={keyInput}
          onChange={(e) => setKeyInput(e.target.value)}
          aria-describedby="settings-bar-help"
        />
      </div>
      <p id="settings-bar-help" className="settings-bar__help">
        Stored in this browser only. The backend uses it for embeddings and chat via OpenRouter (
        <code>openrouter.ai</code>). Never share exports — they omit the key.
      </p>
      <div className="settings-bar__actions">
        <button type="button" className="btn btn--primary" disabled={busy} onClick={save}>
          Save key
        </button>
        <button type="button" className="btn" disabled={busy} onClick={testKey}>
          Test key
        </button>
        <button type="button" className="btn btn--danger" disabled={busy} onClick={clearKey}>
          Clear
        </button>
      </div>
      {msg && <p className="settings-bar__ok">{msg}</p>}
      {err && <p className="inspector__error settings-bar__err">{err}</p>}
    </div>
  );
}
