/**
 * Example usage of RAG services
 * 
 * This file demonstrates how to use the RAG services in a framework-agnostic way.
 * The examples show common use cases for document chunking, embedding, and RAG pipelines.
 */

import { RagService, LlmConfig, VectorStoreConfig } from '../services/rag.service';
import { EmbeddingService } from '../services/embedding.service';
import { TextChunkingService } from '../services/text-chunking.service';
import { LlmProviderEnum } from '../enums/llm-provider.enum';
import { VectorStoreProviderEnum } from '../enums/vector-store-provider.enum';
import { VectorStorageDocumentTypeEnum } from '../enums/vector-storage-document-type.enum';
import { VectorStorageMetadata } from '../interfaces/vector-storage-metadata.interface';
import { Document } from '@langchain/core/documents';

/**
 * Example 1: Basic RAG Service Setup
 * 
 * Initialize the RAG service with OpenAI and Pinecone configuration.
 * API keys should come from environment variables or secure configuration.
 */
export function exampleRagServiceSetup(): RagService {
  const chatLlm: LlmConfig = {
    provider: LlmProviderEnum.OPENAI,
    apiKey: process.env.OPENAI_API_KEY || '',
    model: 'gpt-4o-mini',
  };

  const embeddingLlm: LlmConfig = {
    provider: LlmProviderEnum.OPENAI,
    apiKey: process.env.OPENAI_API_KEY || '',
    model: 'text-embedding-3-small',
  };

  const vectorStoreConfig: VectorStoreConfig = {
    provider: VectorStoreProviderEnum.PINECONE,
    apiKey: process.env.PINECONE_API_KEY || '',
    indexName: process.env.PINECONE_INDEX_NAME || '',
    indexHost: process.env.PINECONE_INDEX_HOST || '',
  };

  return new RagService(chatLlm, embeddingLlm, vectorStoreConfig);
}

/**
 * Example 2: Chunking and Adding Documents to Vector Store
 * 
 * This example shows how to:
 * 1. Chunk a markdown document
 * 2. Convert chunks to LangChain Documents
 * 3. Add them to the vector store
 */
export async function exampleChunkAndAddDocuments() {
  const ragService = exampleRagServiceSetup();
  const chunkingService = new TextChunkingService();

  // Sample markdown content
  const markdownContent = `
# Introduction
This is a sample document about artificial intelligence.

## Machine Learning
Machine learning is a subset of AI that focuses on algorithms.

## Deep Learning
Deep learning uses neural networks with multiple layers.
  `.trim();

  // Create metadata for the document
  const metadata: VectorStorageMetadata = {
    fileName: 'ai-introduction.md',
    extension: 'md',
    fileKey: 's3://bucket/ai-introduction.md',
    type: VectorStorageDocumentTypeEnum.TEXT,
    organizationId: 'org-123',
    workspaceId: 'workspace-456',
    productId: 'product-789',
    createdAt: new Date(),
    documentFileId: 'doc-file-123',
    documentNodeId: 'doc-node-456',
    fileId: 'file-789',
  };

  // Chunk the text
  const chunks = await chunkingService.chunkTextAsDocuments(markdownContent, metadata, {
    chunkSize: 1000,
    overlap: 200,
  });

  // Add to vector store (namespace is typically organizationId)
  const namespace = metadata.organizationId;
  const documentIds = await ragService.addDocuments(namespace, chunks);

  console.log(`Added ${documentIds.length} chunks to vector store`);
  return documentIds;
}

/**
 * Example 3: Creating and Using a RAG Pipeline
 * 
 * This example shows how to:
 * 1. Create a RAG state graph
 * 2. Invoke it with a question
 * 3. Get an answer with context retrieval
 */
export async function exampleRagQuery() {
  const ragService = exampleRagServiceSetup();
  const namespace = 'org-123';

  // Create the RAG state graph with optional custom prompts
  const ragGraph = await ragService.createStateGraph(namespace, {
    corePrompt: 'You are an expert AI assistant specializing in technical documentation.',
    contextPrompt: 'Use the following context to answer the question:',
    historyPrompt: 'Previous conversation summary:',
  });

  // Invoke the graph with a question
  const result = await ragGraph.invoke({
    question: 'What is machine learning?',
    productId: 'product-789',
    summary: 'Previous conversation about AI basics',
  });

  console.log('Answer:', result.answer);
  console.log('Context used:', result.context);
  return result;
}

/**
 * Example 4: Using Embedding Service Directly
 * 
 * This example shows how to use the embedding service independently
 * for custom embedding operations.
 */
export async function exampleEmbeddingService() {
  const embeddingService = new EmbeddingService(process.env.OPENAI_API_KEY || '');

  // Embed a single text
  const singleEmbedding = await embeddingService.embedText('Hello, world!');
  console.log('Single embedding dimension:', singleEmbedding.length);

  // Embed multiple texts
  const texts = ['First document', 'Second document', 'Third document'];
  const multipleEmbeddings = await embeddingService.embedTexts(texts);
  console.log('Multiple embeddings:', multipleEmbeddings.length);

  return { singleEmbedding, multipleEmbeddings };
}

/**
 * Example 5: Document Retrieval by File ID or Name
 * 
 * This example shows how to retrieve documents from the vector store
 * using file identifiers.
 */
export async function exampleDocumentRetrieval() {
  const ragService = exampleRagServiceSetup();
  const namespace = 'org-123';

  // Retrieve documents by file ID
  const documentsByFileId = await ragService.getDocumentsByFileId(namespace, 'file-789', 5);
  console.log(`Found ${documentsByFileId.length} documents by file ID`);

  // Retrieve documents by file name
  const documentsByFileName = await ragService.getDocumentsByFileName(namespace, 'ai-introduction.md', 5);
  console.log(`Found ${documentsByFileName.length} documents by file name`);

  return { documentsByFileId, documentsByFileName };
}

/**
 * Example 6: Removing Documents from Vector Store
 * 
 * This example shows how to remove documents when files are deleted.
 */
export async function exampleRemoveDocuments() {
  const ragService = exampleRagServiceSetup();
  const namespace = 'org-123';

  // Remove all chunks associated with specific file IDs
  const fileIdsToRemove = ['file-789', 'file-123'];
  const success = await ragService.removeDocuments(namespace, fileIdsToRemove);

  console.log(`Removal successful: ${success}`);
  return success;
}

/**
 * Example 7: Text Summarization
 * 
 * This example shows how to summarize multiple texts using the RAG service.
 */
export async function exampleTextSummarization() {
  const ragService = exampleRagServiceSetup();

  const texts = [
    'Artificial intelligence is transforming industries across the globe.',
    'Machine learning algorithms can learn from data without explicit programming.',
    'Deep learning uses neural networks to model complex patterns in data.',
  ];

  const summary = await ragService.summarizeTexts(texts);
  console.log('Summary:', summary);
  return summary;
}

/**
 * Example 8: Complete Workflow - Upload, Chunk, Store, and Query
 * 
 * This example demonstrates a complete workflow from document upload to querying.
 */
export async function exampleCompleteWorkflow() {
  // Step 1: Setup services
  const ragService = exampleRagServiceSetup();
  const chunkingService = new TextChunkingService();
  const namespace = 'org-123';

  // Step 2: Prepare document content and metadata
  const documentContent = `
# Product Documentation

## Features
- Feature 1: Advanced search capabilities
- Feature 2: Real-time updates
- Feature 3: Multi-user collaboration

## Getting Started
To get started, follow these steps:
1. Create an account
2. Set up your workspace
3. Invite team members
  `.trim();

  const metadata: VectorStorageMetadata = {
    fileName: 'product-docs.md',
    extension: 'md',
    fileKey: 's3://bucket/product-docs.md',
    type: VectorStorageDocumentTypeEnum.TEXT,
    organizationId: namespace,
    workspaceId: 'workspace-456',
    productId: 'product-789',
    createdAt: new Date(),
    documentFileId: 'doc-file-123',
    documentNodeId: 'doc-node-456',
    fileId: 'file-789',
  };

  // Step 3: Chunk the document
  const chunks = await chunkingService.chunkTextAsDocuments(documentContent, metadata);

  // Step 4: Add to vector store
  await ragService.addDocuments(namespace, chunks);
  console.log(`Added ${chunks.length} chunks to vector store`);

  // Step 5: Create RAG graph
  const ragGraph = await ragService.createStateGraph(namespace);

  // Step 6: Query the RAG system
  const result = await ragGraph.invoke({
    question: 'What are the main features of the product?',
    productId: 'product-789',
    summary: '',
  });

  console.log('Question:', 'What are the main features of the product?');
  console.log('Answer:', result.answer);

  return result;
}

