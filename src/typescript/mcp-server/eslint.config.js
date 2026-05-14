import eslint from '@eslint/js';
import tseslint from '@typescript-eslint/eslint-plugin';
import tsparser from '@typescript-eslint/parser';
import prettier from 'eslint-config-prettier';

export default [
  eslint.configs.recommended,
  {
    files: ['src/**/*.ts', 'tests/**/*.ts'],
    languageOptions: {
      parser: tsparser,
      parserOptions: {
        ecmaVersion: 2022,
        sourceType: 'module',
        project: './tsconfig.json',
      },
      globals: {
        console: 'readonly',
        process: 'readonly',
        Buffer: 'readonly',
        __dirname: 'readonly',
        __filename: 'readonly',
        module: 'readonly',
        require: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        setInterval: 'readonly',
        clearInterval: 'readonly',
      },
    },
    plugins: {
      '@typescript-eslint': tseslint,
    },
    rules: {
      ...tseslint.configs.recommended.rules,
      ...tseslint.configs['recommended-type-checked'].rules,
      '@typescript-eslint/explicit-function-return-type': 'error',
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/await-thenable': 'error',
      '@typescript-eslint/no-misused-promises': 'error',
      '@typescript-eslint/strict-boolean-expressions': 'warn',
      'no-console': ['warn', { allow: ['error', 'warn'] }],
      // §4.2 path discipline: ban `as CanonicalPath` and `as LocalPath` casts
      // outside src/common/paths.ts. Use fromUserInput/fromValidated/toLocal/
      // toCanonical instead. See docs/specs/16-path-abstraction.md §4.2.
      'no-restricted-syntax': [
        'error',
        {
          selector:
            "TSAsExpression[typeAnnotation.type='TSTypeReference'][typeAnnotation.typeName.name='CanonicalPath']",
          message:
            'Do not cast `as CanonicalPath` directly. Use fromUserInput() or fromValidated() from src/common/paths.ts instead.',
        },
        {
          selector:
            "TSAsExpression[typeAnnotation.type='TSTypeReference'][typeAnnotation.typeName.name='LocalPath']",
          message:
            'Do not cast `as LocalPath` directly. Use toLocal() from src/common/paths.ts instead.',
        },
      ],
    },
  },
  {
    // Allow casts inside paths.ts itself — that is the one place where the
    // brand casts are legitimate (the internal normalizePath result is cast
    // to CanonicalPath, and toLocal identity-map fast-path casts to LocalPath).
    files: ['src/common/paths.ts'],
    rules: {
      'no-restricted-syntax': 'off',
    },
  },
  {
    files: ['tests/**/*.ts'],
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
      'no-console': 'off',
      // Tests for path abstraction use literal `as LocalPath` to construct
      // test fixtures — this is the only permitted test-side cast.
      'no-restricted-syntax': 'off',
    },
  },
  prettier,
  {
    ignores: ['dist/**', 'node_modules/**', '*.js', '*.mjs'],
  },
];
