#!/bin/bash
cd "$(dirname "$0")"
npx tsc && node dist/main.js
