// Copyright (C) 2025 Daniel Mueller <deso@posteo.net>
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

use std::mem::transmute;
use std::ops::AddAssign;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;
use std::ops::MulAssign;

use crate::vector::Vector;


mod private {
  pub trait Sealed {}
}


pub trait Zero
where
  Self: private::Sealed + Sized,
{
  const ZERO: Self;
}

pub trait One
where
  Self: private::Sealed + Sized,
{
  const ONE: Self;
}

macro_rules! impl_zero_one {
  ($ty:ty, $zero:expr, $one:expr) => {
    impl private::Sealed for $ty {}
    impl Zero for $ty {
      const ZERO: $ty = $zero;
    }
    impl One for $ty {
      const ONE: $ty = $one;
    }
  };
}

impl_zero_one!(i8, 0, 1);
impl_zero_one!(i16, 0, 1);
impl_zero_one!(i32, 0, 1);
impl_zero_one!(i64, 0, 1);
impl_zero_one!(isize, 0, 1);
impl_zero_one!(u8, 0, 1);
impl_zero_one!(u16, 0, 1);
impl_zero_one!(u32, 0, 1);
impl_zero_one!(u64, 0, 1);
impl_zero_one!(usize, 0, 1);
impl_zero_one!(f32, 0.0, 1.0);
impl_zero_one!(f64, 0.0, 1.0);


/// A square matrix of dimension N, in column major order.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(C, align(16))]
pub struct Matrix<T, const N: usize>(pub [[T; N]; N]);

impl<T, const N: usize> Matrix<T, N>
where
  T: Copy + Zero,
{
  /// Create a zero matrix.
  #[inline]
  pub const fn zero() -> Self {
    Self([[T::ZERO; N]; N])
  }
}

// TODO: Use const generic `N` once necessary functionality is stable.
impl<T> Matrix<T, 4> {
  #[inline]
  pub fn as_array(&self) -> &[T; 16] {
    // SAFETY: The source and destination types have the same memory layout.
    unsafe { transmute::<&[[T; 4]; 4], &[T; 16]>(&self.0) }
  }

  #[inline]
  pub fn slices_as_array(slf: &[Self]) -> &[[T; 16]] {
    // SAFETY: The source and destination types have the same memory layout.
    unsafe { transmute::<&[Self], &[[T; 16]]>(slf) }
  }
}

impl<T> Matrix<T, 4>
where
  T: Zero + One,
{
  /// Create an identity matrix.
  #[inline]
  #[rustfmt::skip]
  pub const fn identity() -> Self {
    Self([
      [T::ONE,  T::ZERO, T::ZERO, T::ZERO],
      [T::ZERO, T::ONE,  T::ZERO, T::ZERO],
      [T::ZERO, T::ZERO, T::ONE,  T::ZERO],
      [T::ZERO, T::ZERO, T::ZERO, T::ONE],
    ])
  }

  /// Create a translation matrix.
  #[inline]
  #[rustfmt::skip]
  pub const fn translation(x: T, y: T, z: T) -> Self {
    // C.f. https://docs.gl/gl3/glTranslate
    Self([
      [T::ONE,  T::ZERO, T::ZERO, T::ZERO],
      [T::ZERO, T::ONE,  T::ZERO, T::ZERO],
      [T::ZERO, T::ZERO, T::ONE,  T::ZERO],
      [x,       y,       z,       T::ONE],
    ])
  }

  /// Create a scale matrix.
  #[inline]
  #[rustfmt::skip]
  pub fn scale(x: T, y: T, z: T) -> Self {
    // C.f. https://docs.gl/gl3/glScale
    Self([
      [x,       T::ZERO, T::ZERO, T::ZERO],
      [T::ZERO, y,       T::ZERO, T::ZERO],
      [T::ZERO, T::ZERO, z,       T::ZERO],
      [T::ZERO, T::ZERO, T::ZERO, T::ONE],
    ])
  }
}

impl Matrix<f32, 4> {
  /// Create a rotation matrix.
  #[rustfmt::skip]
  pub fn rotation(angle_deg: f32, x: f32, y: f32, z: f32) -> Self {
    // C.f. https://docs.gl/gl3/glRotate
    let angle = angle_deg.to_radians();
    let c = angle.cos();
    let s = angle.sin();
    let [x, y, z] = Vector::<f32, 3>::new(x, y, z).normalized().0;

    Self([
      [x * x * (1.0 - c) + c,       y * x * (1.0 - c) + (z * s), x * z * (1.0 - c) - (y * s), 0.0],
      [x * y * (1.0 - c) - (z * s), y * y * (1.0 - c) + c,       y * z * (1.0 - c) + (x * s), 0.0],
      [x * z * (1.0 - c) + (y * s), y * z * (1.0 - c) - (x * s), z * z * (1.0 - c) + c,       0.0],
      [0.0,                         0.0,                         0.0,                         1.0],
    ])
  }

  /// Create a perspective matrix with the given field of view and
  /// aspect ratio.
  pub fn perspective(fov_y_deg: f32, aspect_ratio: f32, znear: f32, zfar: f32) -> Self {
    // Convert the input to suitable top, bottom, right, and left
    // values of a frustum.
    let top = znear * (0.5 * fov_y_deg).to_radians().tan();
    let bottom = -top;
    let right = top * aspect_ratio;
    let left = -right;

    // Now set up the frustum. This an adaptation of the algorithm that
    // `glFrustum` uses, but for a left-handed coordinate system. C.f.
    // https://docs.gl/gl3/glFrustum
    let a = (right + left) / (right - left);
    let b = (top + bottom) / (top - bottom);
    let c = -(zfar + znear) / (znear - zfar);
    let d = (2.0 * zfar * znear) / (znear - zfar);

    let mut slf = Self::zero();
    slf[0][0] = (2.0 * znear) / (right - left);
    slf[1][1] = (2.0 * znear) / (top - bottom);
    slf[2][0] = a;
    slf[2][1] = b;
    slf[2][2] = c;
    slf[2][3] = 1.0;
    slf[3][2] = d;
    slf
  }

  /// Create an orthographic perspective matrix.
  pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, znear: f32, zfar: f32) -> Self {
    // We follow the same algorithm that `glOrtho` uses. C.f.
    // https://docs.gl/gl3/glOrtho

    let tx = -(right + left) / (right - left);
    let ty = -(top + bottom) / (top - bottom);
    let tz = -(zfar + znear) / (zfar - znear);

    let mut slf = Self::zero();
    slf[0][0] = 2.0 / (right - left);
    slf[1][1] = 2.0 / (top - bottom);
    slf[2][2] = -2.0 / (zfar - znear);
    slf[3][0] = tx;
    slf[3][1] = ty;
    slf[3][2] = tz;
    slf[3][3] = 1.0;
    slf
  }

  /// Create a view matrix looking from `eye` toward `center`.
  #[rustfmt::skip]
  pub fn look_at(eye: Vector<f32, 3>, center: Vector<f32, 3>) -> Self {
    let up = Vector::<f32, 3>::new(0.0, 1.0, 0.0);
    // Based on glm's "look at" logic, slightly adapted:
    // https://github.com/g-truc/glm/blob/2d4c4b4dd31fde06cfffad7915c2b3006402322f/glm/ext/matrix_transform.inl#L175-L196

    let z_axis = (center - eye).normalized();
    let x_axis = up.cross(z_axis).normalized();
    let y_axis = z_axis.cross(x_axis);

    Self([
      [ x_axis.x(),       y_axis.x(),       z_axis.x(),      0.0],
      [ x_axis.y(),       y_axis.y(),       z_axis.y(),      0.0],
      [ x_axis.z(),       y_axis.z(),       z_axis.z(),      0.0],
      [-x_axis.dot(eye), -y_axis.dot(eye), -z_axis.dot(eye), 1.0],
    ])
  }
}

impl<T> Default for Matrix<T, 4>
where
  T: Zero + One,
{
  #[inline]
  fn default() -> Self {
    Self::identity()
  }
}

impl<T, const N: usize> Index<usize> for Matrix<T, N> {
  type Output = [T; N];

  #[inline]
  fn index(&self, idx: usize) -> &Self::Output {
    &self.0[idx]
  }
}

impl<T, const N: usize> IndexMut<usize> for Matrix<T, N> {
  #[inline]
  fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
    &mut self.0[idx]
  }
}

impl<T, const N: usize> Mul<&Self> for Matrix<T, N>
where
  T: Copy + AddAssign<T> + Mul<T, Output = T> + MulAssign<T> + Zero,
{
  type Output = Self;

  fn mul(self, other: &Self) -> Self::Output {
    let mut result = Self::zero();
    for i in 0..N {
      for j in 0..N {
        for k in 0..N {
          result[j][i] += self.0[j][k] * other[k][i];
        }
      }
    }

    result
  }
}

impl<T, const N: usize> Mul<Self> for Matrix<T, N>
where
  T: Copy + AddAssign<T> + Mul<T, Output = T> + MulAssign<T> + Zero,
{
  type Output = Self;

  #[inline]
  fn mul(self, other: Self) -> Self::Output {
    self.mul(&other)
  }
}

impl<T, const N: usize> Mul<&[T; N]> for Matrix<T, N>
where
  T: Copy + AddAssign<T> + Mul<T, Output = T> + MulAssign<T> + Zero,
{
  type Output = [T; N];

  fn mul(self, other: &[T; N]) -> Self::Output {
    let mut result = [T::ZERO; N];
    for (row, r) in result.iter_mut().enumerate() {
      for (col, x) in other.iter().enumerate() {
        *r += self.0[col][row] * *x;
      }
    }

    result
  }
}

impl<T, const N: usize> Mul<&Vector<T, N>> for Matrix<T, N>
where
  T: Copy + AddAssign<T> + Mul<T, Output = T> + MulAssign<T> + Zero,
{
  type Output = Vector<T, N>;

  #[inline]
  fn mul(self, other: &Vector<T, N>) -> Self::Output {
    Vector(self * &other.0)
  }
}

impl<T, const N: usize> Mul<Vector<T, N>> for Matrix<T, N>
where
  T: Copy + AddAssign<T> + Mul<T, Output = T> + MulAssign<T> + Zero,
{
  type Output = Vector<T, N>;

  #[inline]
  fn mul(self, other: Vector<T, N>) -> Self::Output {
    self.mul(&other)
  }
}


#[cfg(test)]
mod tests {
  use super::*;

  #[cfg(feature = "nightly")]
  use std::hint::black_box;

  #[cfg(feature = "nightly")]
  use test::Bencher;


  type Vec4i = Vector<i64, 4>;
  type Mat4i = Matrix<i64, 4>;


  /// Check that we can multiply two matrices.
  #[test]
  fn mat_mul() {
    let m1 = Mat4i::identity();
    let m2 = Mat4i::identity();

    assert_eq!(m1 * m2, m1);

    let m1 = Mat4i::translation(1, 0, 0);
    let m2 = Mat4i::translation(1, 2, 3);
    let expected = Mat4i::translation(2, 2, 3);
    assert_eq!(m1 * m2, expected);
  }


  /// Check that we can multiply a matrix with a vector.
  #[test]
  fn mat_vec_mul() {
    let v = Vec4i::new(1, 2, 3, 1);
    let m = Mat4i::identity();

    assert_eq!(m * v, v);

    let m = Mat4i::translation(1, 0, 0);
    assert_eq!(m * v, Vec4i::new(2, 2, 3, 1));

    let m = Mat4i::translation(0, 1, 0);
    assert_eq!(m * v, Vec4i::new(1, 3, 3, 1));

    let m = Mat4i::translation(0, 0, 1);
    assert_eq!(m * v, Vec4i::new(1, 2, 4, 1));

    let m = Mat4i::translation(3, 2, 1);
    assert_eq!(m * v, Vec4i::new(4, 4, 4, 1));

    let m = Mat4i::scale(10, 1, 1);
    assert_eq!(m * v, Vec4i::new(10, 2, 3, 1));
  }

  /// Benchmark our matrix multiplication logic.
  #[bench]
  #[cfg(feature = "nightly")]
  fn bench_mat_mul(b: &mut Bencher) {
    type Mat4f = Matrix<f32, 4>;

    let m = Mat4f::translation(5.0, 6.0, 7.0);
    let n = Mat4f::translation(1.0, 2.0, 3.0);
    let o = Mat4f::translation(4.0, 3.0, 2.0);

    let () = b.iter(|| {
      let m = black_box(m) * black_box(n) * black_box(o);
      black_box(m);
    });
  }

  /// Benchmark nalgebra matrix multiplication logic.
  #[bench]
  #[cfg(feature = "nightly")]
  fn bench_mat_mul_nalgebra(b: &mut Bencher) {
    use nalgebra::Matrix4;
    use nalgebra::Vector3;

    let m = Matrix4::new_translation(&Vector3::new(5.0, 6.0, 7.0));
    let n = Matrix4::new_translation(&Vector3::new(1.0, 2.0, 3.0));
    let o = Matrix4::new_translation(&Vector3::new(4.0, 3.0, 2.0));

    let () = b.iter(|| {
      let m = black_box(m) * black_box(n) * black_box(o);
      black_box(m);
    });
  }
}
