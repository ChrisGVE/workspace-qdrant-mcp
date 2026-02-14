// TypeScript Declaration File - Regression test for .d.ts detection
// Expected: file_type=code, language=typescript

export interface User {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
  roles: ReadonlyArray<Role>;
}

export type Role = 'admin' | 'editor' | 'viewer';

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

export type AsyncResult<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

export declare function fetchUsers(
  page?: number,
  pageSize?: number,
): Promise<PaginatedResponse<User>>;

export declare function getUserById(id: string): Promise<AsyncResult<User>>;
