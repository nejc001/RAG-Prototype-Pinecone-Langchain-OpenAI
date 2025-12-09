import { VectorStorageDocumentTypeEnum } from '../enums/vector-storage-document-type.enum';

export interface VectorStorageMetadata {
  fileName: string; // original file name
  extension: string;
  fileKey: string; // s3 key, to retrieve the file through the chat
  documentPage?: number;
  type: VectorStorageDocumentTypeEnum;
  startPosition?: number;
  endPosition?: number;
  organizationId: string;
  workspaceId: string;
  productId?: string;
  createdAt: Date;
  documentFileId: string;
  documentNodeId: string;
  fileId: string;
}

