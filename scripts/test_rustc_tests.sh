#!/bin/bash
set -e

cd $(dirname "$0")/../

source ./scripts/setup_rust_fork.sh

echo "[TEST] Test suite of rustc"
pushd rust

cargo install ripgrep

rm -r src/test/ui/{abi/,extern/,panics/,unsized-locals/,thinlto/,simd*,*lto*.rs,linkage*,unwind-*.rs,duplicate/} || true
for test in $(rg --files-with-matches "asm!|catch_unwind|should_panic|lto" src/test/ui); do
  rm $test
done

for test in $(rg --files-with-matches "//~.*ERROR|//~.*NOTE|// error-pattern:|// build-fail" src/test/ui); do
  rm $test
done

git checkout -- src/test/ui/issues/auxiliary/issue-3136-a.rs # contains //~ERROR, but shouldn't be removed

# these all depend on unwinding support
rm src/test/ui/backtrace.rs
rm src/test/ui/intrinsics/intrinsic-move-val-cleanups.rs
rm src/test/ui/array-slice-vec/box-of-array-of-drop-*.rs
rm src/test/ui/array-slice-vec/slice-panic-*.rs
rm src/test/ui/array-slice-vec/nested-vec-3.rs
rm src/test/ui/cleanup-rvalue-temp-during-incomplete-alloc.rs
rm src/test/ui/issues/issue-26655.rs
rm src/test/ui/issues/issue-29485.rs
rm src/test/ui/issues/issue-30018-panic.rs
rm src/test/ui/multi-panic.rs
rm src/test/ui/sepcomp/sepcomp-unwind.rs
rm src/test/ui/structs-enums/unit-like-struct-drop-run.rs
rm src/test/ui/terminate-in-initializer.rs
rm src/test/ui/threads-sendsync/task-stderr.rs
rm src/test/ui/numbers-arithmetic/int-abs-overflow.rs
rm src/test/ui/drop/drop-trait-enum.rs
rm src/test/ui/issues/issue-8460.rs

# these all use ByScalarPair type as extern "C" function parameter => warning
rm src/test/ui/rust-2018/proc-macro-crate-in-paths.rs
rm src/test/ui/proc-macro/crt-static.rs
rm src/test/ui/proc-macro/no-missing-docs.rs
rm src/test/ui/mir/mir_codegen_calls.rs

rm src/test/ui/issues/issue-28950.rs # depends on stack size optimizations
rm src/test/ui/init-large-type.rs # same
rm src/test/ui/sse2.rs # cpuid not supported, so sse2 not detected
rm src/test/ui/issues/issue-33992.rs # unsupported linkages
rm src/test/ui/issues/issue-51947.rs # same
rm src/test/ui/impl-trait/impl-generic-mismatch.rs # same
rm src/test/ui/issues/issue-21160.rs # same
rm src/test/ui/numbers-arithmetic/saturating-float-casts.rs # intrinsic gives different but valid result
rm src/test/ui/mir/mir_misc_casts.rs # depends on deduplication of constants
rm src/test/ui/mir/mir_raw_fat_ptr.rs # same
rm src/test/ui/consts/const-str-ptr.rs # same
rm src/test/ui/async-await/async-fn-size-moved-locals.rs # -Cpanic=abort shrinks some generator by one byte
rm src/test/ui/async-await/async-fn-size-uninit-locals.rs # same
rm src/test/ui/generator/size-moved-locals.rs # same
rm src/test/ui/fn/dyn-fn-alignment.rs # wants a 256 byte alignment
rm src/test/ui/consts/const_in_pattern/issue-73431.rs # gives warning for MY_RUSTC_LOG=warn
rm src/test/ui/test-attrs/test-fn-signature-verification-for-explicit-return-type.rs # "Cannot run dynamic test fn out-of-process"
rm src/test/ui/intrinsics/intrinsic-nearby.rs # unimplemented nearbyintf32 and nearbyintf64 intrinsics

rm src/test/incremental/hashes/inline_asm.rs # inline asm
rm src/test/incremental/issue-72386.rs # same
rm src/test/incremental/change_crate_dep_kind.rs # requires -Cpanic=unwind
rm src/test/incremental/issue-49482.rs # same
rm src/test/incremental/issue-54059.rs # same
rm src/test/incremental/hashes/statics.rs # unsupported linkages
rm src/test/incremental/hashes/function_interfaces.rs # same
rm src/test/incremental/lto.rs # requires lto

rm src/test/pretty/asm.rs # inline asm
rm src/test/pretty/raw-str-nonexpr.rs # same

rm -r src/test/run-pass-valgrind/unsized-locals

echo "[TEST] rustc test suite"
COMPILETEST_FORCE_STAGE0=1 ./x.py test --stage 0 src/test/{codegen-units,run-make,run-pass-valgrind,ui} -vv
popd
