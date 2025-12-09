import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Document } from '@langchain/core/documents';
import { v4 as uuidv4 } from 'uuid';
import { VectorStorageMetadata } from '../interfaces/vector-storage-metadata.interface.js';
import { ChunkingOptions } from '../interfaces/chunking-options.interface.js';
import { DocumentChunk } from '../interfaces/document-chunk.interface.js';
import { ChunkingDefaultsConfig } from '../interfaces/chunking-defaults-config.interface.js';

/**
 * Default chunking configuration constants
 */
const DEFAULT_CHUNKING_CONFIG: Required<ChunkingDefaultsConfig> = {
  chunkSize: 1000,
  overlap: 200,
  separators: ['## ', '# ', '### ', '\n\n', '\n', ' ', ''],
};

/**
 * Framework-agnostic text chunking service.
 * Splits text into chunks with configurable size and overlap.
 */
export class TextChunkingService {
  private readonly defaultConfig: Required<ChunkingDefaultsConfig>;

  constructor(defaultConfig?: ChunkingDefaultsConfig) {
    this.defaultConfig = { ...DEFAULT_CHUNKING_CONFIG, ...defaultConfig };
  }

  /**
   * Chunk text into smaller pieces with metadata
   * @param text - The text to chunk
   * @param metadata - Metadata to attach to each chunk
   * @param options - Chunking options (size, overlap, separators)
   * @returns Promise resolving to array of document chunks
   */
  async chunkText(
    text: string,
    metadata: VectorStorageMetadata,
    options: ChunkingOptions = {}
  ): Promise<DocumentChunk[]> {
    const {
      chunkSize = this.defaultConfig.chunkSize,
      overlap = this.defaultConfig.overlap,
      separators = this.defaultConfig.separators,
    } = options;

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize,
      chunkOverlap: overlap,
      separators,
    });

    const textChunks = await splitter.splitText(text);

    let position = 0;
    const documents: DocumentChunk[] = [];

    for (const chunk of textChunks) {
      const start = text.indexOf(chunk, position);
      const end = start + chunk.length;

      documents.push({
        id: uuidv4(),
        pageContent: chunk,
        metadata: {
          ...metadata,
          startPosition: start,
          endPosition: end,
        },
      });

      position = end;
    }

    return documents;
  }

  /**
   * Chunk text and return as LangChain Document objects
   * @param text - The text to chunk
   * @param metadata - Metadata to attach to each chunk
   * @param options - Chunking options
   * @returns Promise resolving to array of LangChain Document objects
   */
  async chunkTextAsDocuments(
    text: string,
    metadata: VectorStorageMetadata,
    options: ChunkingOptions = {}
  ): Promise<Document<VectorStorageMetadata>[]> {
    const chunks = await this.chunkText(text, metadata, options);

    return chunks.map(
      (chunk) =>
        new Document<VectorStorageMetadata>({
          id: chunk.id,
          pageContent: chunk.pageContent,
          metadata: chunk.metadata,
        })
    );
  }
}

