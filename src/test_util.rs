//! Test-only utility functionality.


macro_rules! assert_eq_f32 {
  ($a:expr, $b:expr) => {
    assert_eq_f32!($a, $b, f32::EPSILON)
  };
  ($a:expr, $b:expr, $epsilon:expr) => {
    // Make sure that we are not comparing against NaN or so, which may
    // behave in unexpected ways.
    assert!(f32::is_finite($a), "{}", $a);
    assert!(f32::is_finite($b), "{}", $b);

    if f32::abs($a - $b) > $epsilon {
      // We basically just forward to the std macro to mirror its error
      // reporting.
      assert_eq!($a, $b);
    }
  };
}


#[cfg(test)]
mod tests {
  use std::hint::black_box;


  /// Check that our float equality passes when equal values are
  /// provided.
  #[test]
  fn f32_equality() {
    assert_eq_f32!(1.0, 1.0);
    assert_eq_f32!(2.0, black_box(1.0) + black_box(1.0));
  }

  /// Check that we can specify an epsilon for our float equality check.
  #[test]
  fn f32_equality_epsilon() {
    let epsilon = 0.6;
    assert_eq_f32!(2.5, 2.0, epsilon);
  }

  /// Check that our float equality check flags inequalities.
  #[test]
  #[should_panic(expected = "assertion `left == right`")]
  fn f32_unequality() {
    assert_eq_f32!(1.0, 1.1);
  }
}
