#!/bin/bash
set -e

cd $(dirname "$0")/../

source build/config.sh

echo "[TEST] Test suite of rustc"
pushd rust

export CG_CLIF_DISPLAY_CG_TIME=0

echo "[TEST] rustc test suite"
RUST_TEST_NOCAPTURE=1 COMPILETEST_FORCE_STAGE0=1 ./x.py test --stage 0 src/test/ui/allocator/not-an-allocator.rs
RUST_TEST_NOCAPTURE=1 COMPILETEST_FORCE_STAGE0=1 ./x.py test --stage 0 src/test/{codegen-units,run-make,run-pass-valgrind,ui}
popd
