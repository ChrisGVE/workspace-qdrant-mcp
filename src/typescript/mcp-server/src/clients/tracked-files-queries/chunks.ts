import type { Database as DatabaseType } from 'better-sqlite3';
import type { DegradedQueryResult } from '../sqlite-state-manager.js';
import { handleTableNotFound } from './helpers.js';

export interface ChunkCandidateEntry {
  pointId: string;
  relativePath: string;
  branch: string | null;
  symbolName: string | null;
  chunkType: string | null;
  startLine: number | null;
}

export interface ListChunkCandidatesOptions {
  watchFolderId: string;
  needles: string[];
  fileType?: string;
  limit?: number;
}

export function listChunkCandidates(
  db: DatabaseType | null,
  options: ListChunkCandidatesOptions
): DegradedQueryResult<ChunkCandidateEntry[]> {
  if (options.needles.length === 0) return { data: [], status: 'ok' };
  if (!db) return { data: [], status: 'degraded', reason: 'database_not_found' };

  try {
    const limit = options.limit ?? 8;
    const conditions = ['tf.watch_folder_id = ?'];
    const params: (string | number)[] = [options.watchFolderId];
    if (options.fileType) {
      conditions.push('tf.file_type = ?');
      params.push(options.fileType);
    }

    const needleClauses: string[] = [];
    for (const needle of options.needles) {
      needleClauses.push('(qc.symbol_name = ? OR tf.relative_path LIKE ?)');
      params.push(needle, `%${needle}%`);
    }
    conditions.push(`(${needleClauses.join(' OR ')})`);
    params.push(limit);

    const rows = db.prepare(`
      SELECT qc.point_id, tf.relative_path, tf.branch, qc.symbol_name, qc.chunk_type, qc.start_line
      FROM qdrant_chunks qc
      JOIN tracked_files tf ON tf.file_id = qc.file_id
      WHERE ${conditions.join(' AND ')}
      ORDER BY
        CASE
          WHEN qc.symbol_name IN (${options.needles.map(() => '?').join(',')}) THEN 0
          ELSE 1
        END,
        tf.relative_path ASC,
        qc.start_line ASC
      LIMIT ?
    `).all(
      ...params.slice(0, -1),
      ...options.needles,
      limit
    ) as Array<{
      point_id: string;
      relative_path: string;
      branch: string | null;
      symbol_name: string | null;
      chunk_type: string | null;
      start_line: number | null;
    }>;

    return {
      data: rows.map((row) => ({
        pointId: row.point_id,
        relativePath: row.relative_path,
        branch: row.branch,
        symbolName: row.symbol_name,
        chunkType: row.chunk_type,
        startLine: row.start_line,
      })),
      status: 'ok',
    };
  } catch (error) {
    return handleTableNotFound(error, [], 'qdrant_chunks');
  }
}
