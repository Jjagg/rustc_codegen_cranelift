#!/bin/bash
set -e

cd $(dirname "$0")/../

source build/config.sh

echo "[TEST] Test suite of rustc"
pushd rust

echo "[TEST] rustc test suite"
COMPILETEST_FORCE_STAGE0=1 ./x.py test --stage 0 src/test/{codegen-units,run-make,run-pass-valgrind,ui}
popd
