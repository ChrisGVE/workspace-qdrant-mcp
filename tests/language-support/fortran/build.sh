#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
gfortran -o bookshelf src/models.f90 src/utils.f90 src/storage.f90 src/main.f90
