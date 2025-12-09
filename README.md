RAG Services (Framework-Agnostic)
=================================

Install & Build
---------------
- Install: `npm install rag-proto`
- Build (emit ESM + types to `dist/`): `npm run build`
- Examples live in `src/examples/rag-usage.example.ts`

What’s inside
-------------
- `services/rag.service.ts`: main RAG orchestration (LLM + vector store + prompts + retrieval + rerank + generation + summarization). Supports pluggable rerankers, MMR fallback, context budgeting, observability hook.
- `services/text-chunking.service.ts`: chunk text with overlap and return chunks or LangChain `Document`s.
- `services/embedding.service.ts`: thin OpenAI embedding client (single + batch).

How the RAG flow works
----------------------
1) **Chunk & embed (outside the service):** use `TextChunkingService` to split raw text; use `EmbeddingService` if you need custom embedding calls.  
2) **Store:** call `RagService.addDocuments(namespace, documents)` to persist LangChain `Document`s into Pinecone (namespace usually an org/workspace ID).  
3) **Retrieve + rerank:** `createStateGraph(namespace, prompts?, options?, reranker?)` builds a LangGraph pipeline:  
   - `retrieve`: similarity search (`similaritySearchWithScore`) against PineconeStore; optional metadata filter.  
   - `rerank`: pluggable reranker interface; fallback to MMR using embeddings; tunable K values.  
   - `generate`: prompt (core/history/context) and LLM answer. Context is budgeted (char budget heuristic) before sending to the LLM.  
4) **Query:** invoke the compiled graph with `{ question, productId, summary }` to get `answer` (and retrieved `context`).  
5) **Maintenance:** `getDocumentsByFileId/Name`, `removeDocuments`, `summarizeTexts` helpers.

What it supports today
----------------------
- LLM provider: OpenAI chat models (`ChatOpenAI`).  
- Embeddings: OpenAI embeddings (`OpenAIEmbeddings`).  
- Vector store: Pinecone via `PineconeStore`, namespaced per tenant.  
- Prompt customization: `corePrompt`, `contextPrompt`, `historyPrompt`.  
- Retrieval config: `retrieveK`, `rerankK`, metadata filter hook, MMR fallback, pluggable reranker interface.  
- Context packing: approximate token budgeting via character budget before generation.  
- Observability: `onEvent` hook for retrieve/generate stages (best-effort).  
- Chunking: configurable size/overlap/separators; returns typed metadata (`VectorStorageMetadata`).

Configuration inputs (secrets expected from env)
------------------------------------------------
- OpenAI: `OPENAI_API_KEY`, chat `model`, embedding `model`.  
- Pinecone: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_INDEX_HOST`.  
- Namespace: caller-provided (e.g., `organizationId`) to isolate tenant data.

Quick start (pseudocode)
------------------------
- Init:
  ```ts
  import { RagService, TextChunkingService } from 'rag-proto';

  const rag = new RagService(chatCfg, embeddingCfg, vectorCfg, options, optionalReranker);
  const chunker = new TextChunkingService();
  ```
- Chunk & store:
  ```ts
  const docs = await chunker.chunkTextAsDocuments(text, metadata);
  await rag.addDocuments(namespace, docs);
  ```
- Ask:
  ```ts
  const graph = await rag.createStateGraph(namespace, prompts, options);
  const res = await graph.invoke({ question, productId, summary: '' });
  console.log(res.answer);
  ```

Next improvements
-----------------
- Add pluggable vector stores (e.g., pgvector/weaviate/qdrant) via adapter interface.  
- Add streaming responses and usage reporting from LLM calls.  
- Add richer telemetry (latency per stage, retrieved IDs/scores, LLM costs).  
- Add guardrails: max tokens per chunk, unsafe content filters, retry/backoff.  
- Extend metadata typing for additional doc types (images/audio) and richer provenance.  
- Provide E2E tests + contract tests for adapters; add lightweight fixtures for examples.  
- Optional: caching for retrieval + prompt/response caching layer.  

