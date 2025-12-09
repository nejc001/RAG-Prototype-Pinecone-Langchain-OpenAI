import { OpenAI } from 'openai';
import { EmbeddingConfig } from '../interfaces/embedding-config.interface.js';

/**
 * Default embedding model constant
 */
const DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small';

/**
 * Framework-agnostic embedding service for generating text embeddings.
 * Currently supports OpenAI embeddings.
 */
export class EmbeddingService {
  private openai: OpenAI;
  private readonly defaultModel: string;

  constructor(apiKey: string, config?: EmbeddingConfig) {
    this.openai = new OpenAI({
      apiKey,
    });
    this.defaultModel = config?.defaultModel ?? DEFAULT_EMBEDDING_MODEL;
  }

  /**
   * Generate embeddings for a text string
   * @param text - The text to embed
   * @param model - The embedding model to use (defaults to configured default model)
   * @returns Promise resolving to the embedding vector
   */
  async embedText(text: string, model?: string): Promise<number[]> {
    const res = await this.openai.embeddings.create({
      input: text,
      model: model ?? this.defaultModel,
    });

    return res.data[0].embedding;
  }

  /**
   * Generate embeddings for multiple text strings
   * @param texts - Array of texts to embed
   * @param model - The embedding model to use (defaults to configured default model)
   * @returns Promise resolving to array of embedding vectors
   */
  async embedTexts(texts: string[], model?: string): Promise<number[][]> {
    const res = await this.openai.embeddings.create({
      input: texts,
      model: model ?? this.defaultModel,
    });

    return res.data.map((item) => item.embedding);
  }
}

