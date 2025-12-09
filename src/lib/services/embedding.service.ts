import { OpenAI } from 'openai';

/**
 * Framework-agnostic embedding service for generating text embeddings.
 * Currently supports OpenAI embeddings.
 */
export class EmbeddingService {
  private openai: OpenAI;

  constructor(apiKey: string) {
    this.openai = new OpenAI({
      apiKey,
    });
  }

  /**
   * Generate embeddings for a text string
   * @param text - The text to embed
   * @param model - The embedding model to use (default: 'text-embedding-3-small')
   * @returns Promise resolving to the embedding vector
   */
  async embedText(text: string, model = 'text-embedding-3-small'): Promise<number[]> {
    const res = await this.openai.embeddings.create({
      input: text,
      model,
    });

    return res.data[0].embedding;
  }

  /**
   * Generate embeddings for multiple text strings
   * @param texts - Array of texts to embed
   * @param model - The embedding model to use (default: 'text-embedding-3-small')
   * @returns Promise resolving to array of embedding vectors
   */
  async embedTexts(texts: string[], model = 'text-embedding-3-small'): Promise<number[][]> {
    const res = await this.openai.embeddings.create({
      input: texts,
      model,
    });

    return res.data.map((item) => item.embedding);
  }
}

