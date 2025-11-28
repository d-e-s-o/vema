// Copyright (C) 2024-2025 Daniel Mueller <deso@posteo.net>
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

#[cfg(test)]
use std::mem::MaybeUninit;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::DivAssign;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;


#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(align(16))]
pub struct Vector<T, const N: usize>(pub [T; N]);

impl<T, const N: usize> Default for Vector<T, N>
where
  T: Copy + Default,
{
  fn default() -> Self {
    Self([T::default(); N])
  }
}

impl<T, const N: usize> Index<usize> for Vector<T, N> {
  type Output = T;

  #[inline]
  fn index(&self, idx: usize) -> &Self::Output {
    &self.0[idx]
  }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
  #[inline]
  fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
    &mut self.0[idx]
  }
}

impl<T> Vector<T, 2> {
  #[inline]
  pub const fn new(x: T, y: T) -> Self {
    Self([x, y])
  }
}

impl<T> Vector<T, 3> {
  #[inline]
  pub const fn new(x: T, y: T, z: T) -> Self {
    Self([x, y, z])
  }

  /// Calculate the dot product between this vector and `other`.
  #[inline]
  pub fn dot(&self, other: Self) -> T
  where
    T: Copy + Add<Output = T> + Mul<Output = T>,
  {
    self[0] * other[0] + self[1] * other[1] + self[2] * other[2]
  }

  /// Calculate the cross product of this vector and `other`.
  #[inline]
  pub fn cross(&self, other: Self) -> Self
  where
    T: Copy + Sub<Output = T> + Mul<Output = T>,
  {
    let x = self[1] * other[2] - self[2] * other[1];
    let y = self[2] * other[0] - self[0] * other[2];
    let z = self[0] * other[1] - self[1] * other[0];
    Self([x, y, z])
  }
}

impl<T> Vector<T, 4> {
  #[inline]
  pub const fn new(x: T, y: T, z: T, w: T) -> Self {
    Self([x, y, z, w])
  }
}

impl<T> Vector<T, 2>
where
  T: Copy,
{
  #[inline]
  pub const fn x(&self) -> T {
    self.0[0]
  }

  #[inline]
  pub const fn y(&self) -> T {
    self.0[1]
  }
}

impl<T> Vector<T, 3>
where
  T: Copy,
{
  #[inline]
  pub const fn x(&self) -> T {
    self.0[0]
  }

  #[inline]
  pub const fn y(&self) -> T {
    self.0[1]
  }

  #[inline]
  pub const fn z(&self) -> T {
    self.0[2]
  }
}

impl<T, const N: usize> Vector<T, N> {
  #[inline]
  pub fn len_squared(&self) -> T
  where
    T: Copy + Add<Output = T> + Mul<Output = T>,
  {
    let (&v, rest) = self.0.split_first().unwrap();
    rest.iter().fold(v * v, |len, &v| len + v * v)
  }
}

impl<const N: usize> Vector<f32, N> {
  #[inline]
  pub fn len(&self) -> f32 {
    let len_sq = self.len_squared();
    len_sq.sqrt()
  }

  #[inline]
  pub fn normalized(&self) -> Self {
    *self / self.len()
  }
}

impl<T> From<Vector<T, 2>> for Vector<T, 3>
where
  T: Copy + Default,
{
  fn from(other: Vector<T, 2>) -> Self {
    Self::new(other.0[0], other.0[1], T::default())
  }
}

impl<T> From<Vector<T, 3>> for Vector<T, 2>
where
  T: Copy,
{
  fn from(other: Vector<T, 3>) -> Self {
    Self::new(other.0[0], other.0[1])
  }
}

impl<T, const N: usize> Vector<T, N> {
  #[cfg(test)]
  #[expect(clippy::multiple_unsafe_ops_per_block)]
  pub fn from_other<U>(other: Vector<U, N>) -> Self
  where
    T: From<U>,
    U: Copy,
  {
    let mut array = MaybeUninit::<[T; N]>::uninit();
    let ptr = array.as_mut_ptr().cast::<T>();
    for i in 0..N {
      // SAFETY: `ptr` points to valid memory of `N` instances of `T`.
      let () = unsafe { ptr.add(i).write(T::from(other[i])) };
    }
    // SAFETY: We just made sure to initialize all members of the
    //         array.
    unsafe { Self(array.assume_init()) }
  }
}

impl<T> From<(T, T, T)> for Vector<T, 3> {
  #[inline]
  fn from(other: (T, T, T)) -> Self {
    Self::new(other.0, other.1, other.2)
  }
}

impl<T, const N: usize> Add<Self> for Vector<T, N>
where
  T: Copy + AddAssign<T>,
  Self: AddAssign,
{
  type Output = Self;

  #[inline]
  fn add(mut self, other: Self) -> Self::Output {
    self += other;
    self
  }
}

impl<T, const N: usize> AddAssign<Self> for Vector<T, N>
where
  T: Copy + AddAssign<T>,
{
  #[inline]
  fn add_assign(&mut self, rhs: Self) {
    for i in 0..N {
      self[i] += rhs[i];
    }
  }
}

impl<T, const N: usize> Sub<Self> for Vector<T, N>
where
  T: Copy + SubAssign<T>,
  Self: SubAssign,
{
  type Output = Self;

  #[inline]
  fn sub(mut self, other: Self) -> Self::Output {
    self -= other;
    self
  }
}

impl<T, const N: usize> SubAssign<Self> for Vector<T, N>
where
  T: Copy + SubAssign<T>,
{
  #[inline]
  fn sub_assign(&mut self, rhs: Self) {
    for i in 0..N {
      self[i] -= rhs[i];
    }
  }
}

impl<T, const N: usize> Mul<T> for Vector<T, N>
where
  T: Copy + MulAssign<T>,
{
  type Output = Self;

  #[inline]
  fn mul(mut self, other: T) -> Self::Output {
    self *= other;
    self
  }
}

impl<T, const N: usize> MulAssign<T> for Vector<T, N>
where
  T: Copy + MulAssign<T>,
{
  #[inline]
  fn mul_assign(&mut self, other: T) {
    for i in 0..N {
      self[i] *= other;
    }
  }
}

impl<T, const N: usize> Div<T> for Vector<T, N>
where
  T: Copy + DivAssign<T>,
{
  type Output = Self;

  #[inline]
  fn div(mut self, other: T) -> Self::Output {
    self /= other;
    self
  }
}

impl<T, const N: usize> DivAssign<T> for Vector<T, N>
where
  T: Copy + DivAssign<T>,
{
  #[inline]
  fn div_assign(&mut self, other: T) {
    for i in 0..N {
      self[i] /= other;
    }
  }
}

impl<T, const N: usize> Neg for Vector<T, N>
where
  T: Copy + Neg<Output = T>,
{
  type Output = Self;

  fn neg(mut self) -> Self::Output {
    for i in 0..N {
      self[i] = -self[i];
    }
    self
  }
}


#[cfg(test)]
mod tests {
  use super::*;


  type VecI = Vector<i64, 3>;
  type VecF = Vector<f32, 3>;


  /// Make sure that we can convert tuples to vectors.
  #[test]
  fn conversion() {
    let v = VecI::from((1, 2, 3));
    assert_eq!(v, VecI::new(1, 2, 3));
  }

  /// Check that the [`Vector::from_other`] constructor works as it
  /// should.
  #[test]
  fn type_conversion() {
    let v = Vector::<u16, 3>::new(1, 2, 3);
    let v = Vector::<u64, 3>::from_other(v);
    assert_eq!(v.x(), 1);
    assert_eq!(v.y(), 2);
    assert_eq!(v.z(), 3);
  }

  /// Check that we can negate a vector.
  #[test]
  fn negation() {
    let v = VecI::new(1, 2, 3);
    assert_eq!(-v, VecI::new(-1, -2, -3));
  }

  /// Test that we can normalize a vector.
  #[test]
  fn normalization() {
    let v1 = VecF::new(5.0, 0.0, 0.0);
    let v2 = v1.normalized();
    assert_eq_f32!(v2.x(), 1.0);
    assert_eq_f32!(v2.y(), 0.0);
    assert_eq_f32!(v2.z(), 0.0);

    let v1 = VecF::new(0.0, 1337.0, 0.0);
    let v2 = v1.normalized();
    assert_eq_f32!(v2.x(), 0.0);
    assert_eq_f32!(v2.y(), 1.0);
    assert_eq_f32!(v2.z(), 0.0);

    let v1 = VecF::new(0.0, 0.0, 20.0);
    let v2 = v1.normalized();
    assert_eq_f32!(v2.x(), 0.0);
    assert_eq_f32!(v2.y(), 0.0);
    assert_eq_f32!(v2.z(), 1.0);
  }

  /// Check that we can calculate the dot product of two vectors.
  #[test]
  fn dot_product() {
    // Two orthogonal vectors.
    let v1 = VecF::new(1.0, 0.0, 0.0);
    let v2 = VecF::new(0.0, 1.0, 0.0);
    assert_eq_f32!(v1.dot(v2), 0.0);
  }

  /// Check that we can calculate the cross product of two vectors.
  #[test]
  fn cross_product() {
    // Two parallel vectors.
    let v1 = VecF::new(1.0, 0.0, 0.0);
    let v2 = VecF::new(-1.0, 0.0, 0.0);
    assert_eq!(v1.cross(v2), VecF::default());
  }

  /// Check that we can add and subtract vectors.
  #[test]
  fn addition_subtraction() {
    let v1 = VecI::new(1, 0, 0);
    let v2 = VecI::new(0, 1, 2);
    let v3 = v1 + v2;
    assert_eq!(v3, VecI::new(1, 1, 2));
    assert_eq!(v3 - v2, v1);
    assert_eq!(v3 - v1, v2);
  }

  /// Check that we can scale a vector by multiplication/division with a
  /// scalar.
  #[test]
  fn scaling() {
    let v1 = VecI::new(1, 2, 3);
    let v2 = v1 * 2;
    assert_eq!(v2, VecI::new(2, 4, 6));
    assert_eq!(v2 / 2, v1);
  }
}
