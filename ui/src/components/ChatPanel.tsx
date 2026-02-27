import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

import type { ChatMessage } from "../types";
import { API, apiRun, chat as apiChat } from "../api";

type Props = {
  inputsDir: string;
  workspace?: string;
  /** Called after a deterministic job completes (e.g., to refresh artifacts/workspace). */
  onJobDone?: () => void;
};

type ActionLabel = "revalidate" | "validate" | "suggest-fix" | "apply-fixes";

const QUICK_ACTIONS: ActionLabel[] = ["revalidate", "validate"];

function inferQuick(text: string): ActionLabel | null {
  const t = (text || "").toLowerCase().trim();
  const token = t.split(/\s+/)[0];
  if (QUICK_ACTIONS.includes(token as ActionLabel)) return token as ActionLabel;

  if (t.includes("revalidate")) return "revalidate";
  if (t.includes("apply") && t.includes("fix")) return "apply-fixes";
  if (t.includes("validate")) return "validate";
  return null;
}

function maybeExplain(text: string): string | null {
  const t = (text || "").toLowerCase();
  const asksValidateVsStress =
    (t.includes("validate") && t.includes("stress")) || t.includes("stress validate");

  if (t.includes("explain") && asksValidateVsStress) {
    return (
      "## Model validate vs stress validate\n\n" +
      "- **Stress validate**: checks your *stress input CSVs* (wells/chd/recharge/etc.) against the grid/idomain and config—missing columns, bad indices, out-of-bounds rows/cols/layers, NaNs, duplicates, and basic sanity checks. It does **not** run MODFLOW.\n" +
      "- **Model validate**: a higher-level preflight that typically *includes stress validation* plus additional checks on the workspace/config.\n\n" +
      "In your current build, **`validate` is an alias for stress validation** to keep UX simple."
    );
  }

  return null;
}

function isNearBottom(el: HTMLElement, thresholdPx = 140) {
  const remaining = el.scrollHeight - el.scrollTop - el.clientHeight;
  return remaining < thresholdPx;
}

function jobOk(job: any) {
  const state = job?.state;
  const exit = job?.exit_code;
  if (state === "error") return { ok: false, exit };
  if (state === "done") {
    if (typeof exit === "number") return { ok: exit === 0, exit };
    return { ok: true, exit };
  }
  return { ok: job?.ok === true, exit };
}

// All questions now route through the unified /chat endpoint.
// The backend LLM has full workspace access (text + binary file probing).

/**
 * If the LLM answer looks like raw JSON/dict, convert it to readable markdown.
 */
function sanitizeAnswerText(raw: string): string {
  const trimmed = (raw || "").trim();
  if (!trimmed) return "No answer returned.";

  // Detect if the answer is raw JSON (starts with { or [)
  if (/^\s*[\[{]/.test(trimmed)) {
    try {
      const parsed = JSON.parse(trimmed);
      // Convert JSON to readable bullet points
      return jsonToMarkdown(parsed);
    } catch {
      // Not valid JSON, might be a Python dict repr — try basic cleanup
      if (/^{['"]/.test(trimmed)) {
        // Looks like a Python dict, convert single quotes to double and try
        try {
          const fixed = trimmed.replace(/'/g, '"');
          const parsed = JSON.parse(fixed);
          return jsonToMarkdown(parsed);
        } catch {
          // Give up, return as-is
        }
      }
    }
  }
  return trimmed;
}

function jsonToMarkdown(obj: any, indent = 0): string {
  if (obj === null || obj === undefined) return "_none_";
  if (typeof obj === "string") return obj;
  if (typeof obj === "number" || typeof obj === "boolean") return String(obj);

  if (Array.isArray(obj)) {
    if (obj.length === 0) return "_empty list_";
    // If array of simple values, inline them
    if (obj.every((v) => typeof v !== "object" || v === null)) {
      return obj.map((v) => `- ${v}`).join("\n");
    }
    // Array of objects: render each as a bullet with sub-items
    return obj
      .map((item, i) => {
        if (typeof item === "object" && item !== null) {
          const entries = Object.entries(item)
            .map(([k, v]) => `**${k}:** ${typeof v === "object" ? JSON.stringify(v) : v}`)
            .join(" | ");
          return `- ${entries}`;
        }
        return `- ${item}`;
      })
      .join("\n");
  }

  if (typeof obj === "object") {
    const entries = Object.entries(obj);
    if (entries.length === 0) return "_empty_";
    return entries
      .map(([key, val]) => {
        const label = key.replace(/_/g, " ");
        if (Array.isArray(val)) {
          return `**${label}:**\n${jsonToMarkdown(val, indent + 1)}`;
        }
        if (typeof val === "object" && val !== null) {
          return `**${label}:**\n${jsonToMarkdown(val, indent + 1)}`;
        }
        return `**${label}:** ${val}`;
      })
      .join("\n\n");
  }

  return String(obj);
}

// formatWorkspaceAnswer removed — unified /chat endpoint returns reply directly.

/** Prefix relative URLs with the API base so images/links resolve correctly. */
function resolveApiUrl(url?: string): string | undefined {
  if (!url) return url;
  if (url.startsWith("http") || url.startsWith("data:")) return url;
  return `${API}${url.startsWith("/") ? "" : "/"}${url}`;
}

/** Check if a URL points to a plot output endpoint (image). */
function isPlotOutputUrl(url: string): boolean {
  return url.includes("/plots/run/output");
}

// ── Deterministic chips: run via /run endpoint, no LLM ──
const DETERMINISTIC_CHIPS = [
  { label: "Validate", action: "validate" as ActionLabel, icon: "✓" },
  { label: "Revalidate", action: "revalidate" as ActionLabel, icon: "↻" },
];

// ── AI-powered chips: either call the LLM chat or an LLM-backed CLI command ──
type AiChip =
  | { label: string; icon: string; action: ActionLabel; isRunQuick: true; message?: undefined }
  | { label: string; icon: string; message: string; action?: undefined; isRunQuick?: undefined };

const AI_CHIPS: AiChip[] = [
  // Suggest Fix — routed through the LLM /chat endpoint so it can use conversation
  // context (e.g. a preceding QA review) to produce actionable fix suggestions.
  {
    label: "Suggest Fix",
    icon: "🔧",
    message: "Based on the QA findings above, suggest specific fixes for this MODFLOW model. For each issue, describe what to change, which files or packages are affected, and the expected impact.",
  },
  // QA / analysis chips — send a pre-written message to the LLM /chat endpoint
  {
    label: "QA Overview",
    icon: "🔍",
    message: "Run a comprehensive QA/QC review of this model. Check mass balance, dry cells, convergence, and property values. Flag any issues and provide recommendations.",
  },
  {
    label: "Mass Balance",
    icon: "⚖️",
    message: "Analyze the mass balance for this model. Show percent discrepancy by stress period and flag any periods with poor balance.",
  },
  {
    label: "Dry Cells",
    icon: "🏜️",
    message: "Check for dry cells in the model. Show counts by layer and time step, identify spatial clusters, and note any trends.",
  },
  {
    label: "Pumping Review",
    icon: "💧",
    message: "Review the pumping data. Show rates by stress period, check for anomalous rate changes, and generate a cumulative pumping summary.",
  },
  {
    label: "Model Summary",
    icon: "📋",
    message: "Provide a comprehensive summary of this MODFLOW model: grid dimensions, layers, packages, stress periods, boundary conditions, and solver settings.",
  },
  {
    label: "Calibration Tips",
    icon: "🎯",
    message: "Based on the model setup and any available run results, suggest calibration improvements. What parameters should be adjusted first and what are typical calibration targets?",
  },
];

const DEFAULT_GREETING: ChatMessage = {
  role: "assistant",
  content: "GW Copilot is ready. Ask questions about your model, run validation, or use the quick actions below.",
};

const CHAT_STORAGE_PREFIX = "gw_copilot_chat:";
const MAX_PERSISTED_MESSAGES = 200;

function chatStorageKey(inputsDir: string): string {
  return `${CHAT_STORAGE_PREFIX}${inputsDir}`;
}

function loadChatHistory(inputsDir: string): ChatMessage[] {
  try {
    const raw = localStorage.getItem(chatStorageKey(inputsDir));
    if (!raw) return [DEFAULT_GREETING];
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.length > 0) return parsed;
  } catch {}
  return [DEFAULT_GREETING];
}

function saveChatHistory(inputsDir: string, messages: ChatMessage[]) {
  try {
    // Keep only the last N messages to prevent localStorage bloat
    const toSave = messages.slice(-MAX_PERSISTED_MESSAGES);
    localStorage.setItem(chatStorageKey(inputsDir), JSON.stringify(toSave));
  } catch {}
}

export default function ChatPanel({ inputsDir, workspace, onJobDone }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadChatHistory(inputsDir));
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [lastLog, setLastLog] = useState<string>("");
  const [actionError, setActionError] = useState<string>("");

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const shouldStickToBottomRef = useRef(true);
  const prevInputsDirRef = useRef(inputsDir);

  // When project changes, save current chat and load the new project's chat
  useEffect(() => {
    if (prevInputsDirRef.current !== inputsDir) {
      // Save the old project's chat
      saveChatHistory(prevInputsDirRef.current, messages);
      // Load the new project's chat
      setMessages(loadChatHistory(inputsDir));
      prevInputsDirRef.current = inputsDir;
    }
  }, [inputsDir]);

  // Persist chat on every message change
  useEffect(() => {
    if (inputsDir) {
      saveChatHistory(inputsDir, messages);
    }
  }, [messages, inputsDir]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => { shouldStickToBottomRef.current = isNearBottom(el); };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (!shouldStickToBottomRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, thinking, lastLog]);

  async function runQuick(action: ActionLabel) {
    setLastLog("");
    setActionError("");
    setThinking(true);
    try {
      const res = await apiRun(action, { inputs_dir: inputsDir, workspace: workspace ?? null }, (line) => {
        setLastLog((prev) => (prev ? prev + "\n" + line : line));
      });
      const { ok, exit } = jobOk(res);
      let resultMsg: string;
      if (ok) {
        resultMsg = `\u2705 **${action}** passed \u2014 no errors found.`;
      } else if (exit === 2) {
        resultMsg = `\u26a0\ufe0f **${action}** found issues. Check the **Artifacts** tab for the full report.`;
      } else {
        resultMsg = `\u274c **${action}** failed unexpectedly (exit code ${exit ?? "unknown"}). Check logs for details.`;
      }
      setMessages((m) => [
        ...m,
        { role: "assistant", content: resultMsg },
      ]);
      if (typeof onJobDone === "function") onJobDone();
    } catch (e: any) {
      const msg = e?.message ?? String(e);
      setActionError(msg);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `**${action}** failed: ${msg}`,
        },
      ]);
    } finally {
      setThinking(false);
      setLastLog("");
    }
  }

  async function send(text: string) {
    const trimmed = (text || "").trim();
    if (!trimmed || thinking) return;

    setMessages((m) => [...m, { role: "user", content: trimmed }]);
    setInput("");
    setThinking(true);
    setActionError("");

    try {
      const explain = maybeExplain(trimmed);
      if (explain) {
        setMessages((m) => [...m, { role: "assistant", content: explain }]);
        return;
      }

      const quick = inferQuick(trimmed);
      if (quick && quick !== "apply-fixes") {
        await runQuick(quick);
        return;
      }

      // All questions route through the unified /chat endpoint.
      // The backend has full workspace access (text files + binary probing via FloPy).
      const history = messages.map((m) => ({ role: m.role, content: m.content }));
      const res = await apiChat({
        message: trimmed,
        inputs_dir: inputsDir,
        workspace: workspace ?? null,
        history,
      });

      const reply = sanitizeAnswerText(res.reply || "");
      setMessages((m) => [...m, { role: "assistant", content: reply }]);

      if (res.suggestions?.length) {
        setMessages((m) => [
          ...m,
          { role: "assistant", content: `**Suggestions:** ${res.suggestions.join(" | ")}` },
        ]);
      }
    } catch (e: any) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Something went wrong: ${e?.message ?? String(e)}` },
      ]);
    } finally {
      setThinking(false);
    }
  }

  return (
    <div className="chat-panel">
      {/* Messages area */}
      <div className="chat-scroll" ref={scrollRef}>
        <div className="chat-thread">
          {messages.map((m, idx) => (
            <div key={idx} className={`chat-row ${m.role}`}>
              <div className="chat-role-label">
                {m.role === "user" ? "You" : "⚡ Copilot"}
              </div>
              <div className={`chat-bubble ${m.role}`}>
                <ReactMarkdown
                  components={{
                    img: ({ src, alt, ...props }) => {
                      const resolvedSrc = resolveApiUrl(src);
                      return (
                        <img
                          src={resolvedSrc}
                          alt={alt || "Plot"}
                          style={{
                            maxWidth: "100%",
                            borderRadius: 6,
                            marginTop: 8,
                            marginBottom: 8,
                            border: "1px solid var(--border, #ddd)",
                          }}
                          {...props}
                        />
                      );
                    },
                    a: ({ href, children, ...props }) => {
                      // If link points to a plot output, render as an inline image
                      if (href && isPlotOutputUrl(href)) {
                        const resolvedSrc = resolveApiUrl(href);
                        return (
                          <img
                            src={resolvedSrc}
                            alt={String(children ?? "Plot")}
                            style={{
                              maxWidth: "100%",
                              borderRadius: 6,
                              marginTop: 8,
                              marginBottom: 8,
                              border: "1px solid var(--border, #ddd)",
                            }}
                          />
                        );
                      }
                      return (
                        <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
                          {children}
                        </a>
                      );
                    },
                  }}
                >
                  {m.content}
                </ReactMarkdown>
              </div>
            </div>
          ))}

          {lastLog ? (
            <div className="chat-row assistant">
              <div className="chat-role-label">⚡ Copilot</div>
              <div className="chat-bubble assistant">
                <div className="chat-log-label">Live output:</div>
                <pre className="chat-log-pre">{lastLog}</pre>
              </div>
            </div>
          ) : null}

          {thinking && !lastLog && (
            <div className="chat-row assistant">
              <div className="chat-role-label">⚡ Copilot</div>
              <div className="chat-bubble assistant chat-thinking-bubble">
                <span className="chat-dot" /><span className="chat-dot" /><span className="chat-dot" />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Composer */}
      <div className="chat-footer">
        {/* Row 1: Deterministic actions (no LLM) */}
        <div className="chat-chips">
          <span className="chat-chips-label chat-chips-label-det">{"\u2699\ufe0f"} DETERMINISTIC</span>
          {DETERMINISTIC_CHIPS.map((chip) => (
            <button
              key={chip.action}
              className="chat-chip chat-chip-det"
              onClick={() => {
                setMessages((m) => [...m, { role: "user", content: chip.label }]);
                runQuick(chip.action);
              }}
              disabled={thinking}
              title={`Run ${chip.label} (deterministic — no AI)`}
            >
              <span className="chat-chip-icon">{chip.icon}</span>
              {chip.label}
            </button>
          ))}
          {messages.length > 1 && (
            <button
              className="chat-chip"
              onClick={() => setMessages([DEFAULT_GREETING])}
              disabled={thinking}
              title="Clear chat history"
              style={{ marginLeft: "auto" }}
            >
              <span className="chat-chip-icon">&#x2715;</span>
              Clear Chat
            </button>
          )}
        </div>

        {/* Row 2: AI-powered actions (use LLM) */}
        <div className="chat-chips chat-chips-ai-row">
          <span className="chat-chips-label chat-chips-label-ai">{"\u2728"} AI-POWERED</span>
          {AI_CHIPS.map((chip) => (
            <button
              key={chip.label}
              className="chat-chip chat-chip-ai"
              onClick={() => {
                if (chip.isRunQuick && chip.action) {
                  setMessages((m) => [...m, { role: "user", content: chip.label }]);
                  runQuick(chip.action);
                } else if (chip.message) {
                  send(chip.message);
                }
              }}
              disabled={thinking}
              title={chip.message ?? `Run ${chip.label} (AI-powered)`}
            >
              <span className="chat-chip-icon">{chip.icon}</span>
              {chip.label}
            </button>
          ))}
        </div>

        <div className="chat-compose-row">
          <input
            className="chat-compose-input"
            placeholder="Ask about your model or type a command…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
            disabled={thinking}
          />
          <button
            className="btn primary chat-send-btn"
            onClick={() => send(input)}
            disabled={thinking || !input.trim()}
          >
            {thinking ? "…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
