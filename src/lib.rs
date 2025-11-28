// Copyright (C) 2025 Daniel Mueller <deso@posteo.net>
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

#![cfg_attr(feature = "nightly", feature(test))]

#[cfg(feature = "nightly")]
extern crate test;

#[cfg(test)]
#[macro_use]
mod test_util;

mod matrix;
mod vector;

pub use crate::matrix::Matrix;
pub use crate::matrix::One;
pub use crate::matrix::Zero;
pub use crate::vector::Vector;
