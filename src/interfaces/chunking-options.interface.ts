import { LlmProviderEnum } from '../enums/llm-provider.enum.js';

/**
 * Options for text chunking operations
 */
export interface ChunkingOptions {
  chunkSize?: number;
  overlap?: number;
  separators?: string[];
  llmModel?: LlmProviderEnum;
  embeddingModel?: string;
}

