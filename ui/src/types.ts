export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface RunResponse {
  ok: boolean;
  exit_code: number;
  stdout: string;
  stderr: string;
  artifacts: Record<string, string>;
}
