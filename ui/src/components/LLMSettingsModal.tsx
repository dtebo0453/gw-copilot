import React, { useEffect, useState } from "react";

const API_BASE =
  (import.meta as any).env?.VITE_API_BASE || "";

type Props = {
  onClose: () => void;
};

type LLMConfig = {
  provider: "openai" | "anthropic";
  openai_api_key?: string;
  anthropic_api_key?: string;
  model?: string;
};

async function fetchJson(url: string, init?: RequestInit) {
  const r = await fetch(url, init);
  const txt = await r.text();
  let data: any = null;
  try {
    data = txt ? JSON.parse(txt) : null;
  } catch {}
  if (!r.ok) {
    const detail = (data && (data.detail || data.message)) || txt || `HTTP ${r.status}`;
    throw new Error(String(detail));
  }
  return data;
}

export function LLMSettingsModal({ onClose }: Props) {
  const [provider, setProvider] = useState<"openai" | "anthropic">("openai");
  const [openaiKey, setOpenaiKey] = useState("");
  const [anthropicKey, setAnthropicKey] = useState("");
  const [model, setModel] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [status, setStatus] = useState<any>(null);

  // Load current config on mount
  useEffect(() => {
    fetchJson(`${API_BASE}/llm/config`)
      .then((cfg) => {
        if (cfg) {
          setProvider(cfg.provider || "openai");
          setModel(cfg.model || "");
          // Don't load actual keys for security - just show if configured
          setStatus(cfg);
        }
      })
      .catch(() => {
        // Ignore - endpoint might not exist yet
      });
  }, []);

  async function handleSave() {
    setSaving(true);
    setError(null);
    setSuccess(false);

    try {
      const config: LLMConfig = {
        provider,
        model: model || undefined,
      };

      // Always send both keys if entered so switching providers doesn't lose keys
      if (openaiKey.trim()) {
        config.openai_api_key = openaiKey.trim();
      }
      if (anthropicKey.trim()) {
        config.anthropic_api_key = anthropicKey.trim();
      }

      await fetchJson(`${API_BASE}/llm/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });

      setSuccess(true);
      setTimeout(() => onClose(), 1500);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setSaving(false);
    }
  }

  const openaiModels = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"];
  const anthropicModels = ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"];

  return (
    <div className="modalOverlay" onClick={onClose}>
      <div className="modal llm-settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modalTitle">LLM Settings</div>
        <div className="modalBody">
          <div className="llm-settings-section">
            <label className="label">Provider</label>
            <div className="llm-provider-toggle">
              <button
                className={`llm-provider-btn ${provider === "openai" ? "active" : ""}`}
                onClick={() => { setProvider("openai"); setModel(""); }}
              >
                <span className="llm-provider-icon">🤖</span>
                OpenAI
              </button>
              <button
                className={`llm-provider-btn ${provider === "anthropic" ? "active" : ""}`}
                onClick={() => { setProvider("anthropic"); setModel(""); }}
              >
                <span className="llm-provider-icon">🧠</span>
                Claude (Anthropic)
              </button>
            </div>
          </div>

          <div className="llm-settings-section">
            <label className="label">OpenAI API Key</label>
            <input
              type="password"
              className="input"
              value={openaiKey}
              onChange={(e) => setOpenaiKey(e.target.value)}
              placeholder={status?.openai_configured ? "••••••••••••••••" : "sk-..."}
            />
            {status?.openai_configured && !openaiKey && (
              <div className="muted">API key already configured. Enter a new key to change it.</div>
            )}
          </div>

          <div className="llm-settings-section">
            <label className="label">Anthropic API Key</label>
            <input
              type="password"
              className="input"
              value={anthropicKey}
              onChange={(e) => setAnthropicKey(e.target.value)}
              placeholder={status?.anthropic_configured ? "••••••••••••••••" : "sk-ant-..."}
            />
            {status?.anthropic_configured && !anthropicKey && (
              <div className="muted">API key already configured. Enter a new key to change it.</div>
            )}
          </div>

          <div className="llm-settings-section">
            <label className="label">Model</label>
            <select
              className="input"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="">Default</option>
              {(provider === "openai" ? openaiModels : anthropicModels).map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          {error && <div className="llm-settings-error">{error}</div>}
          {success && <div className="llm-settings-success">Settings saved successfully!</div>}
        </div>

        <div className="modalActions">
          <button className="btn" onClick={onClose} disabled={saving}>
            Cancel
          </button>
          <button className="btn primary" onClick={handleSave} disabled={saving}>
            {saving ? "Saving..." : "Save Settings"}
          </button>
        </div>
      </div>
    </div>
  );
}
