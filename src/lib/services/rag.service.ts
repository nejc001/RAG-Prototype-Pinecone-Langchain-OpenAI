import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { Index, Pinecone, RecordMetadata } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { Document, DocumentInterface } from '@langchain/core/documents';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { Annotation, CompiledStateGraph, StateGraph } from '@langchain/langgraph';
import { MessageContentComplex } from '@langchain/core/messages';
import { loadSummarizationChain } from 'langchain/chains';
import { LlmProviderEnum } from '../enums/llm-provider.enum';
import { VectorStoreProviderEnum } from '../enums/vector-store-provider.enum';
import { VectorStorageMetadata } from '../interfaces/vector-storage-metadata.interface';
import { cosineSimilarity } from '../utils/vector.util';

/**
 * Configuration for LLM provider
 */
export interface LlmConfig {
  provider: LlmProviderEnum;
  apiKey: string;
  model: string;
}

/**
 * Configuration for vector store
 */
export interface VectorStoreConfig {
  provider: VectorStoreProviderEnum;
  apiKey: string;
  indexName: string;
  indexHost: string;
}

/**
 * Optional prompts for RAG state graph
 */
export interface RagPrompts {
  corePrompt?: string;
  contextPrompt?: string;
  historyPrompt?: string;
}

/**
 * Optional hooks and knobs for the RAG pipeline
 */
export interface RagOptions {
  /** Retrieve top K from vector store (pre-rerank) */
  retrieveK?: number;
  /** Rerank to top K after reranker */
  rerankK?: number;
  /** Simple character budget for context packing (approximate token budget) */
  contextCharBudget?: number;
  /** Enable MMR-based rerank when no external reranker is provided */
  enableMmrRerank?: boolean;
  /** MMR lambda (balance relevance vs diversity) */
  mmrLambda?: number;
  /** Observability hook */
  onEvent?: (event: { stage: string; data?: unknown }) => void;
  /** Custom metadata filter before rerank */
  filter?: (doc: DocumentInterface) => boolean;
}

/**
 * Reranker interface (plug in cross-encoder / API-based rerankers)
 */
export interface Reranker {
  rerank: (
    query: string,
    candidates: Array<{ doc: DocumentInterface; score?: number }>
  ) => Promise<DocumentInterface[]>;
}

/**
 * Framework-agnostic RAG (Retrieval-Augmented Generation) service.
 * Provides functionality for creating RAG pipelines, managing vector stores,
 * and performing document retrieval and generation.
 */
export class RagService {
  private chatLLM!: LlmProviderEnum;
  private embeddingsLLM!: LlmProviderEnum;
  private vectorStoreProvider!: VectorStoreProviderEnum;

  private llm!: ChatOpenAI;
  private embeddings!: OpenAIEmbeddings;
  private pinecone!: Pinecone;
  private pineconeIndex!: Index<RecordMetadata>;
  private vectorStores: Map<string, PineconeStore> = new Map();
  private readonly options: RagOptions;
  private readonly reranker?: Reranker;

  constructor(
    chatLlm: LlmConfig,
    embeddingLlm: LlmConfig,
    vectorStoreConfig: VectorStoreConfig,
    options: RagOptions = {},
    reranker?: Reranker
  ) {
    if (chatLlm.provider !== LlmProviderEnum.OPENAI) {
      throw new Error(`Unsupported LLM provider: ${chatLlm.provider}`);
    }

    this.llm = new ChatOpenAI({
      model: chatLlm.model,
      apiKey: chatLlm.apiKey,
      temperature: 0.2,
      topP: 1,
      frequencyPenalty: 0,
      presencePenalty: 0,
      n: 1,
      stopSequences: ['\n\nHuman:', '\n\nAssistant:'],
    });

    if (embeddingLlm.provider !== LlmProviderEnum.OPENAI) {
      throw new Error(`Unsupported embedding provider: ${embeddingLlm.provider}`);
    }

    this.embeddings = new OpenAIEmbeddings({
      openAIApiKey: embeddingLlm.apiKey,
      model: embeddingLlm.model,
    });

    if (vectorStoreConfig.provider !== VectorStoreProviderEnum.PINECONE) {
      throw new Error(`Unsupported vector store provider: ${vectorStoreConfig.provider}`);
    }

    this.pinecone = new Pinecone({ apiKey: vectorStoreConfig.apiKey });
    this.pineconeIndex = this.pinecone.Index(vectorStoreConfig.indexName, vectorStoreConfig.indexHost);

    this.options = {
      retrieveK: 12,
      rerankK: 5,
      contextCharBudget: 12_000, // ~3 chars/token heuristic
      enableMmrRerank: true,
      mmrLambda: 0.5,
      ...options,
    };
    this.reranker = reranker;
  }

  /**
   * Create a state graph for RAG pipeline
   * @param namespace - Namespace for the vector store (typically organizationId or knowledge base identifier)
   * @param optionalPrompts - Optional custom prompts for the RAG pipeline
   * @returns Compiled state graph ready to invoke
   */
  public async createStateGraph(
    namespace: string,
    optionalPrompts?: RagPrompts
  ): Promise<CompiledStateGraph<any, any, any, any, any, any>> {
    const promptTemplate = ChatPromptTemplate.fromMessages([
      [
        'system',
        `${optionalPrompts?.corePrompt ?? 'You are a helpful assistant.'}

${optionalPrompts?.historyPrompt ?? 'Here might be a history of chat of what the user and you have previously discussed:'}
{summary}

${optionalPrompts?.contextPrompt ?? 'Here is context you might find useful:'}
{context}`,
      ],
      ['human', `{question}`],
    ]);

    const InputStateAnnotation = Annotation.Root({
      question: Annotation<string>,
      productId: Annotation<string>,
    });

    const StateAnnotation = Annotation.Root({
      question: Annotation<string>,
      productId: Annotation<string>,
      context: Annotation<DocumentInterface[]>,
      summary: Annotation<string>,
      answer: Annotation<string | MessageContentComplex[]>,
    });

    const retrieve = async (state: typeof InputStateAnnotation.State): Promise<{ context: DocumentInterface[] }> => {
      const vectorStore = this.getOrCreateNamespaceVectorStore(namespace);
      this.emit('retrieve:start', { namespace, query: state.question, productId: state.productId });

      const retrievedDocsWithScores = await vectorStore.similaritySearchWithScore(
        state.question,
        this.options.retrieveK ?? 12,
        {
          productId: state.productId,
        }
      );

      const filtered = this.options.filter
        ? retrievedDocsWithScores.filter(([doc]) => this.options.filter!(doc))
        : retrievedDocsWithScores;

      const rerankedDocs = await this.rerank(state.question, filtered);
      const topContext = rerankedDocs.slice(0, this.options.rerankK ?? 5);

      this.emit('retrieve:done', { retrieved: filtered.length, used: topContext.length });
      return { context: topContext };
    };

    const packContextWithBudget = (docs: DocumentInterface[]): string => {
      const budget = this.options.contextCharBudget ?? 12_000;
      const pieces: string[] = [];
      let used = 0;
      for (const d of docs) {
        const chunk = d.pageContent ?? '';
        if (!chunk) continue;
        if (used + chunk.length > budget) break;
        pieces.push(chunk);
        used += chunk.length;
      }
      return pieces.join('\n');
    };

    const generate = async (
      state: typeof StateAnnotation.State
    ): Promise<{
      answer: string | MessageContentComplex[];
    }> => {
      this.emit('generate:start', { namespace, query: state.question, contextCount: state.context.length });

      const docsContent = packContextWithBudget(state.context);
      const messages = await promptTemplate.invoke({
        question: state.question,
        productId: state.productId,
        context: docsContent,
        summary: state.summary,
      });
      const response = await this.llm.invoke(messages);
      this.emit('generate:done', { length: String(response.content || '').length });
      return { answer: response.content };
    };

    return new StateGraph(StateAnnotation)
      .addNode('retrieve', retrieve)
      .addNode('generate', generate)
      .addEdge('__start__', 'retrieve')
      .addEdge('retrieve', 'generate')
      .addEdge('generate', '__end__')
      .compile();
  }

  /**
   * Add documents to the vector store
   * @param namespace - Namespace for the vector store
   * @param documents - Array of LangChain Document objects to add
   * @returns Promise resolving to array of document IDs
   */
  public async addDocuments(namespace: string, documents: Document[]): Promise<string[]> {
    const vectorStore = this.getOrCreateNamespaceVectorStore(namespace);
    return vectorStore.addDocuments(documents, { namespace });
  }

  /**
   * Remove documents from the vector store by file IDs
   * @param namespace - Namespace for the vector store
   * @param fileIds - Array of file IDs to remove
   * @returns Promise resolving to true if successful
   * @throws Error if deletion fails
   */
  public async removeDocuments(namespace: string, fileIds: string[]): Promise<boolean> {
    const vectorStore = this.getOrCreateNamespaceVectorStore(namespace);

    try {
      const idsToDelete: string[] = [];

      for (const fileId of fileIds) {
        const documents = await this.getDocumentsByFileId(namespace, fileId, 10);
        idsToDelete.push(...documents.map((doc) => doc.id || '').filter(Boolean));
      }

      if (idsToDelete.length > 0) {
        await vectorStore.delete({
          ids: idsToDelete,
          namespace,
        });
      }

      return true;
    } catch (error) {
      throw new Error(`Failed to remove documents: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Get documents by file name using similarity search
   * @param namespace - Namespace for the vector store
   * @param fileName - File name to search for
   * @param topK - Number of results to return
   * @returns Promise resolving to array of documents
   */
  public async getDocumentsByFileName(namespace: string, fileName: string, topK: number): Promise<Document[]> {
    const vectorStore = this.getOrCreateNamespaceVectorStore(namespace);
    const embedding = await this.embeddings.embedQuery(fileName);

    const results = await vectorStore.similaritySearchVectorWithScore(embedding, topK, {
      fileName,
    });

    return results.map(([doc]) => doc);
  }

  /**
   * Get documents by file ID using similarity search
   * @param namespace - Namespace for the vector store
   * @param fileId - File ID to search for
   * @param topK - Number of results to return
   * @returns Promise resolving to array of documents
   */
  public async getDocumentsByFileId(namespace: string, fileId: string, topK: number): Promise<Document[]> {
    const vectorStore = this.getOrCreateNamespaceVectorStore(namespace);
    const embedding = await this.embeddings.embedQuery(fileId);

    const results = await vectorStore.similaritySearchVectorWithScore(embedding, topK, {
      fileId,
    });

    return results.map(([doc]) => doc);
  }

  /**
   * Summarize multiple texts using map-reduce summarization chain
   * @param texts - Array of texts to summarize
   * @returns Promise resolving to summarized text
   */
  public async summarizeTexts(texts: string[]): Promise<string> {
    const docs = texts.map((t) => new Document({ pageContent: t }));

    const chain = loadSummarizationChain(this.llm, { type: 'map_reduce' });

    const result = await chain.invoke({ input_documents: docs });
    return result?.['output_text'] || '';
  }

  /**
   * Get or create a vector store for a given namespace
   * @param namespace - Namespace identifier
   * @returns PineconeStore instance
   */
  private getOrCreateNamespaceVectorStore(namespace: string): PineconeStore {
    const existing = this.vectorStores.get(namespace);
    if (existing) {
      return existing;
    }

    const created = new PineconeStore(this.embeddings, {
      pineconeIndex: this.pineconeIndex,
      namespace,
    });
    this.vectorStores.set(namespace, created);
    return created;
  }

  /**
   * Emit observability events if hook is provided
   */
  private emit(stage: string, data?: unknown): void {
    try {
      this.options.onEvent?.({ stage, data });
    } catch {
      // best-effort, do not fail the pipeline
    }
  }

  /**
   * Rerank documents using provided reranker, fallback to MMR, then to score sort.
   */
  private async rerank(
    query: string,
    candidates: Array<[DocumentInterface, number | undefined]>
  ): Promise<DocumentInterface[]> {
    if (candidates.length === 0) return [];

    // External reranker
    if (this.reranker) {
      const docs = await this.reranker.rerank(
        query,
        candidates.map(([doc, score]) => ({ doc, score }))
      );
      return docs;
    }

    // MMR fallback using embeddings
    if (this.options.enableMmrRerank) {
      const rerankK = this.options.rerankK ?? 5;
      const lambda = this.options.mmrLambda ?? 0.5;
      const queryEmbedding = await this.embeddings.embedQuery(query);

      const docsWithEmbeddings = await Promise.all(
        candidates.map(async ([doc, score]) => ({
          doc,
          score: score ?? 0,
          embedding: await this.embeddings.embedQuery(doc.pageContent || ''),
        }))
      );

      const selected: typeof docsWithEmbeddings = [];
      const remaining = [...docsWithEmbeddings];

      while (selected.length < rerankK && remaining.length > 0) {
        let bestIdx = 0;
        let bestScore = -Infinity;
        for (let i = 0; i < remaining.length; i++) {
          const cand = remaining[i];
          const relevance = cosineSimilarity(queryEmbedding, cand.embedding);
          const diversity =
            selected.length === 0
              ? 0
              : Math.max(...selected.map((s) => cosineSimilarity(s.embedding, cand.embedding)));
          const mmrScore = lambda * relevance - (1 - lambda) * diversity;
          if (mmrScore > bestScore) {
            bestScore = mmrScore;
            bestIdx = i;
          }
        }
        selected.push(remaining[bestIdx]);
        remaining.splice(bestIdx, 1);
      }

      return selected.map((s) => s.doc);
    }

    // Fallback: simple score sort
    return candidates
      .sort(([, scoreA], [, scoreB]) => (scoreB ?? 0) - (scoreA ?? 0))
      .map(([doc]) => doc);
  }
}

