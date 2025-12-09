/**
 * Configuration for default prompts used in RAG pipeline
 */
export interface DefaultPromptsConfig {
  /** Default core prompt for the assistant. Default: 'You are a helpful assistant.' */
  corePrompt?: string;
  /** Default context prompt. Default: 'Here is context you might find useful:' */
  contextPrompt?: string;
  /** Default history prompt. Default: 'Here might be a history of chat of what the user and you have previously discussed:' */
  historyPrompt?: string;
}
