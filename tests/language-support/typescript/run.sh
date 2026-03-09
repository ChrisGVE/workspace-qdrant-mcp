#!/bin/bash
cd "$(dirname "$0")"
npm install --ignore-scripts >/dev/null 2>&1
npx tsc >/dev/null 2>&1 && node dist/main.js
