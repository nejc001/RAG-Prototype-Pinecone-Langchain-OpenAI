// Utils
export { removeNullishProperties, sortObjectKeys, normalizeEmail } from './lib/utils/general.util';

// Services
export { sendTextMail, sendEmailsWithDynamicTemplate, sendEmailWithDynamicTemplate } from './lib/services/sendgrid-mailer.service';

// RAG Services
export { RagService } from './lib/services/rag.service';
export { EmbeddingService } from './lib/services/embedding.service';
export { TextChunkingService } from './lib/services/text-chunking.service';

// Interfaces
export type { BaseEntityInterface } from './lib/interfaces/base-entity.interface';
export type { PaginationOptions } from './lib/interfaces/paginated-options.interface';
export type { PaginatedResult } from './lib/interfaces/paginated-result.interface';
export type { PaginatedResponse } from './lib/interfaces/paginated-response.interface';
export type { VectorStorageMetadata } from './lib/interfaces/vector-storage-metadata.interface';
export type { DocumentChunk } from './lib/interfaces/document-chunk.interface';
export type { ChunkingOptions } from './lib/interfaces/chunking-options.interface';

// RAG Service Types
export type { LlmConfig, VectorStoreConfig, RagPrompts } from './lib/services/rag.service';

// Enums
export { CloudStorageProviderEnum } from './lib/enums/cloud-storage-provider.enum';
export { LlmProviderEnum } from './lib/enums/llm-provider.enum';
export { VectorStoreProviderEnum } from './lib/enums/vector-store-provider.enum';
export { VectorStorageDocumentTypeEnum } from './lib/enums/vector-storage-document-type.enum';
