import { VectorStorageMetadata } from './vector-storage-metadata.interface.js';

/**
 * Interface representing a document chunk with its content and metadata.
 * This is a framework-agnostic representation of a document chunk.
 */
export interface DocumentChunk {
  id: string;
  pageContent: string;
  metadata: VectorStorageMetadata;
}

