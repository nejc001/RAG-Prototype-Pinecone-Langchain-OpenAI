/**
 * Configuration for default chunking parameters
 */
export interface ChunkingDefaultsConfig {
  /** Default chunk size. Default: 1000 */
  chunkSize?: number;
  /** Default overlap between chunks. Default: 200 */
  overlap?: number;
  /** Default separators for recursive text splitting. Default: ['## ', '# ', '### ', '\n\n', '\n', ' ', ''] */
  separators?: string[];
}
