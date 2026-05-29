/**
 * Tests for request-context: the getEffectiveCwd resolution chain and the
 * body-cwd override decision that backs the tool `cwd` argument.
 *
 * Precedence for project detection: header > body `cwd` > WQM_DEFAULT_HOST_CWD
 * > process.cwd().
 */

import { describe, it, expect, afterEach } from 'vitest';
import {
  runWithRequestContext,
  getEffectiveCwd,
  resolveBodyCwdOverride,
} from '../../src/utils/request-context.js';

describe('request-context', () => {
  const savedEnv = process.env.WQM_DEFAULT_HOST_CWD;
  afterEach(() => {
    if (savedEnv === undefined) {
      delete process.env.WQM_DEFAULT_HOST_CWD;
    } else {
      process.env.WQM_DEFAULT_HOST_CWD = savedEnv;
    }
  });

  describe('getEffectiveCwd', () => {
    it('prefers the request-context host cwd (header) above env and process.cwd', () => {
      process.env.WQM_DEFAULT_HOST_CWD = '/env/default';
      const got = runWithRequestContext({ hostCwd: '/from/header' }, () => getEffectiveCwd());
      expect(got).toBe('/from/header');
    });

    it('falls back to WQM_DEFAULT_HOST_CWD when no header is bound', () => {
      process.env.WQM_DEFAULT_HOST_CWD = '/env/default';
      expect(getEffectiveCwd()).toBe('/env/default');
    });

    it('falls back to process.cwd() when neither header nor env is set', () => {
      delete process.env.WQM_DEFAULT_HOST_CWD;
      expect(getEffectiveCwd()).toBe(process.cwd());
    });
  });

  describe('resolveBodyCwdOverride', () => {
    it('returns the body cwd when no header host cwd is bound', () => {
      expect(resolveBodyCwdOverride('C:\\Users\\x\\proj')).toBe('C:\\Users\\x\\proj');
    });

    it('returns undefined (header wins) when a header host cwd is bound', () => {
      const got = runWithRequestContext({ hostCwd: '/from/header' }, () =>
        resolveBodyCwdOverride('C:\\Users\\x\\proj')
      );
      expect(got).toBeUndefined();
    });

    it('returns undefined for an empty or missing body cwd', () => {
      expect(resolveBodyCwdOverride(undefined)).toBeUndefined();
      expect(resolveBodyCwdOverride('')).toBeUndefined();
    });
  });
});
