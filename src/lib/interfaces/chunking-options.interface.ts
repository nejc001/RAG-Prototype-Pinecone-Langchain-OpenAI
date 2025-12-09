import { LlmProviderEnum } from '../enums/llm-provider.enum';

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

