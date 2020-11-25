//! Argument passing

use crate::prelude::*;

use cranelift_codegen::ir::ArgumentPurpose;
use rustc_target::abi::call::{ArgAbi, PassMode as AbiPassMode};
pub(super) use EmptySinglePair::*;

#[derive(Copy, Clone, Debug)]
pub(super) enum PassMode {
    NoPass,
    ByVal(Type),
    ByValPair(Type, Type),
    ByRef { sized: bool, on_stack: Option<Size> },
}

#[derive(Copy, Clone, Debug)]
pub(super) enum EmptySinglePair<T> {
    Empty,
    Single(T),
    Pair(T, T),
}

impl<T> EmptySinglePair<T> {
    pub(super) fn into_iter(self) -> EmptySinglePairIter<T> {
        EmptySinglePairIter(self)
    }

    pub(super) fn map<U>(self, mut f: impl FnMut(T) -> U) -> EmptySinglePair<U> {
        match self {
            Empty => Empty,
            Single(v) => Single(f(v)),
            Pair(a, b) => Pair(f(a), f(b)),
        }
    }
}

pub(super) struct EmptySinglePairIter<T>(EmptySinglePair<T>);

impl<T> Iterator for EmptySinglePairIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match std::mem::replace(&mut self.0, Empty) {
            Empty => None,
            Single(v) => Some(v),
            Pair(a, b) => {
                self.0 = Single(b);
                Some(a)
            }
        }
    }
}

impl<T: std::fmt::Debug> EmptySinglePair<T> {
    pub(super) fn assert_single(self) -> T {
        match self {
            Single(v) => v,
            _ => panic!("Called assert_single on {:?}", self),
        }
    }

    pub(super) fn assert_pair(self) -> (T, T) {
        match self {
            Pair(a, b) => (a, b),
            _ => panic!("Called assert_pair on {:?}", self),
        }
    }
}

impl PassMode {
    pub(super) fn get_param_ty(self, tcx: TyCtxt<'_>) -> EmptySinglePair<Type> {
        match self {
            PassMode::NoPass => Empty,
            PassMode::ByVal(clif_type) => Single(clif_type),
            PassMode::ByValPair(a, b) => Pair(a, b),
            PassMode::ByRef {
                sized: true,
                on_stack: _,
            } => Single(pointer_ty(tcx)),
            PassMode::ByRef {
                sized: false,
                on_stack: _,
            } => Pair(pointer_ty(tcx), pointer_ty(tcx)),
        }
    }

    pub(super) fn get_abi_param(self, tcx: TyCtxt<'_>) -> EmptySinglePair<AbiParam> {
        match self {
            PassMode::NoPass => Empty,
            PassMode::ByVal(clif_type) => Single(AbiParam::new(clif_type)),
            PassMode::ByValPair(a, b) => Pair(AbiParam::new(a), AbiParam::new(b)),
            PassMode::ByRef {
                sized: true,
                on_stack: None,
            } => Single(AbiParam::new(pointer_ty(tcx))),
            PassMode::ByRef {
                sized: true,
                on_stack: Some(size),
            } => {
                let purpose = ArgumentPurpose::StructArgument(
                    u32::try_from(size.bytes()).expect("struct too big to pass on stack"),
                );
                Single(AbiParam::special(pointer_ty(tcx), purpose))
            }
            PassMode::ByRef {
                sized: false,
                on_stack,
            } => {
                assert!(on_stack.is_none());
                Pair(
                    AbiParam::new(pointer_ty(tcx)),
                    AbiParam::new(pointer_ty(tcx)),
                )
            }
        }
    }
}

pub(super) fn get_pass_mode<'tcx>(
    tcx: TyCtxt<'tcx>,
    maybe_struct_arg: bool,
    layout: TyAndLayout<'tcx>,
) -> PassMode {
    if layout.is_zst() {
        // WARNING zst arguments must never be passed, as that will break CastKind::ClosureFnPointer
        PassMode::NoPass
    } else {
        match &layout.abi {
            Abi::Uninhabited => PassMode::NoPass,
            Abi::Scalar(scalar) => PassMode::ByVal(scalar_to_clif_type(tcx, scalar.clone())),
            Abi::ScalarPair(a, b) => {
                let a = scalar_to_clif_type(tcx, a.clone());
                let b = scalar_to_clif_type(tcx, b.clone());
                if a == types::I128 && b == types::I128 {
                    // Returning (i128, i128) by-val-pair would take 4 regs, while only 3 are
                    // available on x86_64. Cranelift gets confused when too many return params
                    // are used.
                    PassMode::ByRef {
                        sized: true,
                        on_stack: None,
                    }
                } else {
                    PassMode::ByValPair(a, b)
                }
            }

            // FIXME implement Vector Abi in a cg_llvm compatible way
            Abi::Vector { .. } => {
                if let Some(vector_ty) = crate::intrinsics::clif_vector_type(tcx, layout) {
                    PassMode::ByVal(vector_ty)
                } else {
                    PassMode::ByRef {
                        sized: true,
                        on_stack: None,
                    }
                }
            }

            Abi::Aggregate { sized: true } => PassMode::ByRef {
                sized: true,
                on_stack: if maybe_struct_arg {
                    Some(layout.size)
                } else {
                    None
                },
            },
            Abi::Aggregate { sized: false } => PassMode::ByRef {
                sized: false,
                on_stack: None,
            },
        }
    }
}

/// Get a set of values to be passed as function arguments.
pub(super) fn adjust_arg_for_abi<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    maybe_struct_arg: bool,
    arg: CValue<'tcx>,
) -> EmptySinglePair<Value> {
    match get_pass_mode(fx.tcx, maybe_struct_arg, arg.layout()) {
        PassMode::NoPass => Empty,
        PassMode::ByVal(_) => Single(arg.load_scalar(fx)),
        PassMode::ByValPair(_, _) => {
            let (a, b) = arg.load_scalar_pair(fx);
            Pair(a, b)
        }
        PassMode::ByRef {
            sized: _,
            on_stack: _,
        } => match arg.force_stack(fx) {
            (ptr, None) => Single(ptr.get_addr(fx)),
            (ptr, Some(meta)) => Pair(ptr.get_addr(fx), meta),
        },
    }
}

/// Create a [`CValue`] containing the value of a function parameter adding clif function parameters
/// as necessary.
pub(super) fn cvalue_for_param<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    maybe_struct_arg: bool,
    start_block: Block,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))] local: Option<mir::Local>,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))] local_field: Option<usize>,
    arg_ty: Ty<'tcx>,
) -> Option<CValue<'tcx>> {
    let layout = fx.layout_of(arg_ty);
    let pass_mode = get_pass_mode(fx.tcx, maybe_struct_arg, layout);

    if let PassMode::NoPass = pass_mode {
        return None;
    }

    let clif_types = pass_mode.get_param_ty(fx.tcx);
    let block_params = clif_types.map(|t| fx.bcx.append_block_param(start_block, t));

    #[cfg(debug_assertions)]
    crate::abi::comments::add_arg_comment(
        fx,
        "arg",
        local,
        local_field,
        block_params,
        pass_mode,
        arg_ty,
    );

    match pass_mode {
        PassMode::NoPass => unreachable!(),
        PassMode::ByVal(_) => Some(CValue::by_val(block_params.assert_single(), layout)),
        PassMode::ByValPair(_, _) => {
            let (a, b) = block_params.assert_pair();
            Some(CValue::by_val_pair(a, b, layout))
        }
        PassMode::ByRef {
            sized: true,
            on_stack: _,
        } => Some(CValue::by_ref(
            Pointer::new(block_params.assert_single()),
            layout,
        )),
        PassMode::ByRef {
            sized: false,
            on_stack: _,
        } => {
            let (ptr, meta) = block_params.assert_pair();
            Some(CValue::by_ref_unsized(Pointer::new(ptr), meta, layout))
        }
    }
}

pub(super) fn arg_abi_to_pass_mode<'tcx>(
    tcx: TyCtxt<'tcx>,
    arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
) -> PassMode {
    // FIXME respect arg extension mode
    match arg_abi.mode {
        AbiPassMode::Ignore => PassMode::NoPass,
        AbiPassMode::Direct(_) => match &arg_abi.layout.abi {
            Abi::Scalar(scalar) => PassMode::ByVal(scalar_to_clif_type(tcx, scalar.clone())),
            _ => unreachable!(),
        },
        AbiPassMode::Pair(_, _) => match &arg_abi.layout.abi {
            Abi::ScalarPair(a, b) => {
                let a = scalar_to_clif_type(tcx, a.clone());
                let b = scalar_to_clif_type(tcx, b.clone());
                if a == types::I128 && b == types::I128 {
                    // Returning (i128, i128) by-val-pair would take 4 regs, while only 3 are
                    // available on x86_64. Cranelift gets confused when too many return params
                    // are used.
                    PassMode::ByRef {
                        sized: true,
                        on_stack: None,
                    }
                } else {
                    PassMode::ByValPair(a, b)
                }
            }
            _ => unreachable!(),
        },
        // FIXME implement this
        AbiPassMode::Cast(_) => PassMode::ByRef {
            sized: true,
            on_stack: None,
        },
        AbiPassMode::Indirect {
            attrs: _,
            extra_attrs: None,
            on_stack,
        } => PassMode::ByRef {
            sized: true,
            on_stack: if on_stack {
                Some(arg_abi.layout.size)
            } else {
                None
            },
        },
        AbiPassMode::Indirect {
            attrs: _,
            extra_attrs: Some(_),
            on_stack,
        } => PassMode::ByRef {
            sized: false,
            on_stack: None,
        },
    }
}
