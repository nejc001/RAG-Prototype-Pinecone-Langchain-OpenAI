/**
 * Configuration for LLM parameters
 */
export interface LlmParamsConfig {
  /** Temperature for sampling (0-2). Higher values make output more random. Default: 0.2 */
  temperature?: number;
  /** Top-p (nucleus) sampling parameter. Default: 1 */
  topP?: number;
  /** Frequency penalty (-2.0 to 2.0). Default: 0 */
  frequencyPenalty?: number;
  /** Presence penalty (-2.0 to 2.0). Default: 0 */
  presencePenalty?: number;
  /** Number of completions to generate. Default: 1 */
  n?: number;
  /** Stop sequences to end generation. Default: ['\n\nHuman:', '\n\nAssistant:'] */
  stopSequences?: string[];
}
