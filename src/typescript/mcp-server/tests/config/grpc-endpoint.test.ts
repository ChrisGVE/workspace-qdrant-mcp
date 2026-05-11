import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { parseGrpcEndpoint, loadConfig } from '../../src/config.js';

describe('parseGrpcEndpoint', () => {
  it('parses http://host:port correctly', () => {
    expect(parseGrpcEndpoint('http://memexd:50051')).toEqual({ host: 'memexd', port: 50051 });
  });

  it('parses bare host:port correctly', () => {
    expect(parseGrpcEndpoint('memexd:50051')).toEqual({ host: 'memexd', port: 50051 });
  });

  it('parses https://host:port correctly', () => {
    expect(parseGrpcEndpoint('https://daemon:9999')).toEqual({ host: 'daemon', port: 9999 });
  });

  it('parses bare host without port (defaults to 50051)', () => {
    expect(parseGrpcEndpoint('myhost')).toEqual({ host: 'myhost', port: 50051 });
  });

  it('parses http://host:port with non-standard port', () => {
    expect(parseGrpcEndpoint('http://daemon:9999')).toEqual({ host: 'daemon', port: 9999 });
  });
});

describe('loadConfig — WQM_DAEMON_ENDPOINT env var', () => {
  const envKeys = ['WQM_DAEMON_ENDPOINT', 'MEMEXD_GRPC_URL', 'WQM_DAEMON_PORT'];
  const saved: Record<string, string | undefined> = {};

  beforeEach(() => {
    for (const k of envKeys) {
      saved[k] = process.env[k];
      delete process.env[k];
    }
  });

  afterEach(() => {
    for (const k of envKeys) {
      if (saved[k] === undefined) {
        delete process.env[k];
      } else {
        process.env[k] = saved[k];
      }
    }
  });

  it('WQM_DAEMON_ENDPOINT sets grpcHost and grpcPort', () => {
    process.env['WQM_DAEMON_ENDPOINT'] = 'http://daemon:9999';
    const config = loadConfig();
    expect(config.daemon.grpcHost).toBe('daemon');
    expect(config.daemon.grpcPort).toBe(9999);
  });

  it('MEMEXD_GRPC_URL alias sets grpcHost and grpcPort', () => {
    process.env['MEMEXD_GRPC_URL'] = 'http://memexd:50051';
    const config = loadConfig();
    expect(config.daemon.grpcHost).toBe('memexd');
    expect(config.daemon.grpcPort).toBe(50051);
  });

  it('WQM_DAEMON_ENDPOINT takes precedence over MEMEXD_GRPC_URL', () => {
    process.env['WQM_DAEMON_ENDPOINT'] = 'preferred:1111';
    process.env['MEMEXD_GRPC_URL'] = 'http://other:2222';
    const config = loadConfig();
    expect(config.daemon.grpcHost).toBe('preferred');
    expect(config.daemon.grpcPort).toBe(1111);
  });

  it('defaults to localhost:50051 when no endpoint env vars set', () => {
    const config = loadConfig();
    expect(config.daemon.grpcHost).toBe('localhost');
    expect(config.daemon.grpcPort).toBe(50051);
  });
});
