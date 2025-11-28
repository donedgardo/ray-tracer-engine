use crate::coord::Coord;
use crate::float_eq::f32_are_eq;
use crate::point::Point;
use crate::tuple::Tuple;
use crate::vector::Vector;
use std::ops::{Index, IndexMut, Mul};

#[derive(Debug, Clone, Copy)]
pub struct Matrix4x4 {
    data: [[f32; 4]; 4],
}

impl Matrix4x4 {
    pub fn new(r1: [f32; 4], r2: [f32; 4], r3: [f32; 4], r4: [f32; 4]) -> Self {
        Self {
            data: [r1, r2, r3, r4],
        }
    }
    pub fn identity() -> Self {
        Self::new(
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        )
    }

    pub fn transpose(&self) -> Self {
        Self::new(
            [
                self.data[0][0],
                self.data[1][0],
                self.data[2][0],
                self.data[3][0],
            ],
            [
                self.data[0][1],
                self.data[1][1],
                self.data[2][1],
                self.data[3][1],
            ],
            [
                self.data[0][2],
                self.data[1][2],
                self.data[2][2],
                self.data[3][2],
            ],
            [
                self.data[0][3],
                self.data[1][3],
                self.data[2][3],
                self.data[3][3],
            ],
        )
    }

    pub fn submatrix(&self, row: usize, col: usize) -> Matrix3x3 {
        let mut data = Vec::with_capacity(3 * 3);
        for y in 0..=3 {
            for x in 0..=3 {
                if y != row && x != col {
                    data.push(self.data[y][x]);
                }
            }
        }
        Matrix3x3::new(
            [data[0], data[1], data[2]],
            [data[3], data[4], data[5]],
            [data[6], data[7], data[8]],
        )
    }
    pub fn minor(&self, row: usize, col: usize) -> f32 {
        self.submatrix(row, col).determinant()
    }

    pub fn cofactor(&self, row: usize, col: usize) -> f32 {
        if (row + col) % 2 != 0 {
            return -self.minor(row, col);
        }
        self.minor(row, col)
    }

    pub fn determinant(&self) -> f32 {
        let mut det = 0.;
        for col in 0..4 {
            det += self.data[0][col] * self.cofactor(0, col);
        }
        det
    }

    pub fn is_invertible(&self) -> bool {
        self.determinant() != 0.
    }

    pub fn inverse(&self) -> Option<Self> {
        if !self.is_invertible() {
            return None;
        }
        let mut inverse = Self::identity();
        let determinant = self.determinant();
        for row in 0..4 {
            for col in 0..4 {
                let cofactor = self.cofactor(row, col);
                inverse.data[col][row] = cofactor / determinant;
            }
        }
        return Some(inverse);
    }
}

impl PartialEq for Matrix4x4 {
    fn eq(&self, other: &Self) -> bool {
        matrices_are_equal(self.data.iter(), other.data.iter())
    }
}

impl Index<usize> for Matrix4x4 {
    type Output = [f32; 4];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Matrix4x4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Mul for Matrix4x4 {
    type Output = Matrix4x4;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result: [[f32; 4]; 4] = [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ];
        for row in 0..=3 {
            for col in 0..=3 {
                result[row][col] = self.data[row][0] * rhs.data[0][col]
                    + self.data[row][1] * rhs.data[1][col]
                    + self.data[row][2] * rhs.data[2][col]
                    + self.data[row][3] * rhs.data[3][col]
            }
        }
        Self::new(result[0], result[1], result[2], result[3])
    }
}

impl Mul<Tuple> for Matrix4x4 {
    type Output = Tuple;

    fn mul(self, rhs: Tuple) -> Self::Output {
        let mut result: [f32; 4] = [0., 0., 0., 0.];
        for row in 0..=3 {
            result[row] = self.data[row][0] * rhs.x()
                + self.data[row][1] * rhs.y()
                + self.data[row][2] * rhs.z()
                + self.data[row][3] * rhs.w()
        }
        Tuple::new(result[0], result[1], result[2], result[3])
    }
}

impl Mul<Point> for Matrix4x4 {
    type Output = Point;

    fn mul(self, rhs: Point) -> Self::Output {
        Point::from(self * Tuple::from(rhs))
    }
}

impl Mul<Matrix4x4> for Point {
    type Output = Point;

    fn mul(self, rhs: Matrix4x4) -> Self::Output {
        Point::from(rhs * Tuple::from(self))
    }
}

impl Mul<Vector> for Matrix4x4 {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Self::Output {
        rhs
    }
}

struct Translation;

impl Translation {
    pub fn new(x: f32, y: f32, z: f32) -> Matrix4x4 {
        let mut tranlation = Matrix4x4::identity();
        tranlation.data[0][3] = x;
        tranlation.data[1][3] = y;
        tranlation.data[2][3] = z;
        tranlation
    }
}

struct Scaling(Matrix4x4);

impl Scaling {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        let mut m = Matrix4x4::identity();
        m[0][0] = x;
        m[1][1] = y;
        m[2][2] = z;
        Self(m)
    }

    pub fn inverse(&self) -> Option<Self> {
        match self.0.inverse() {
            None => None,
            Some(inverse) => Some(Self(inverse)),
        }
    }
}

impl Mul<Point> for Scaling {
    type Output = Point;

    fn mul(self, rhs: Point) -> Self::Output {
        Point::from(self.0 * rhs)
    }
}

impl Mul<Vector> for Scaling {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Self::Output {
        Vector::from(self.0 * Tuple::from(rhs))
    }
}

#[derive(Debug)]
pub struct Matrix3x3 {
    data: [[f32; 3]; 3],
}

impl Matrix3x3 {
    pub fn new(r1: [f32; 3], r2: [f32; 3], r3: [f32; 3]) -> Self {
        Self { data: [r1, r2, r3] }
    }

    pub fn submatrix(&self, row: usize, col: usize) -> Matrix2x2 {
        let mut data = Vec::with_capacity(4);
        for y in 0..=2 {
            for x in 0..=2 {
                if y != row && x != col {
                    data.push(self.data[y][x]);
                }
            }
        }
        Matrix2x2::new([data[0], data[1]], [data[2], data[3]])
    }

    pub fn minor(&self, row: usize, col: usize) -> f32 {
        self.submatrix(row, col).determinant()
    }

    pub fn cofactor(&self, row: usize, col: usize) -> f32 {
        if (row + col) % 2 != 0 {
            return -self.minor(row, col);
        }
        self.minor(row, col)
    }

    pub fn determinant(&self) -> f32 {
        let mut det = 0.;
        for col in 0..3 {
            det += self.data[0][col] * self.cofactor(0, col);
        }
        det
    }
}

impl PartialEq for Matrix3x3 {
    fn eq(&self, other: &Self) -> bool {
        matrices_are_equal(self.data.iter(), other.data.iter())
    }
}

impl Index<usize> for Matrix3x3 {
    type Output = [f32; 3];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

#[derive(Debug)]
pub struct Matrix2x2 {
    data: [[f32; 2]; 2],
}

impl Matrix2x2 {
    pub fn new(r1: [f32; 2], r2: [f32; 2]) -> Self {
        Self { data: [r1, r2] }
    }
    pub fn determinant(&self) -> f32 {
        self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
    }
}

impl PartialEq for Matrix2x2 {
    fn eq(&self, other: &Self) -> bool {
        matrices_are_equal(self.data.iter(), other.data.iter())
    }
}

impl Index<usize> for Matrix2x2 {
    type Output = [f32; 2];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

fn matrices_are_equal<T>(data: T, other: T) -> bool
where
    T: IntoIterator,
    T::Item: AsRef<[f32]>,
{
    data.into_iter().zip(other).all(|(row1, row2)| {
        row1.as_ref()
            .iter()
            .zip(row2.as_ref().iter())
            .all(|(x, y)| f32_are_eq(*x, *y))
    })
}

#[cfg(test)]
mod matrix_tests {
    use crate::matrix::{Matrix2x2, Matrix3x3, Matrix4x4};
    use rstest::rstest;

    #[rstest]
    #[case((0, 0), 1.)]
    #[case((0, 3), 4.)]
    #[case((1, 0), 5.5)]
    #[case((1, 2), 7.5)]
    #[case((2, 2), 11.)]
    #[case((3, 0), 13.5)]
    #[case((3, 2), 15.5)]
    fn constructing_and_inspection_4x4(#[case] index: (usize, usize), #[case] expected: f32) {
        let m = Matrix4x4::new(
            [1., 2., 3., 4.],
            [5.5, 6.5, 7.5, 8.5],
            [9., 10., 11., 12.],
            [13.5, 14.5, 15.5, 16.5],
        );
        assert_eq!(m[index.0][index.1], expected);
    }

    #[rstest]
    #[case((0, 0), -3.)]
    #[case((0, 1), 5.)]
    #[case((1, 0), 1.)]
    #[case((1, 1), -2.)]
    fn constructing_and_inspection_2x2(#[case] index: (usize, usize), #[case] expected: f32) {
        let m = Matrix2x2::new([-3., 5.], [1., -2.]);
        assert_eq!(m[index.0][index.1], expected);
    }

    #[rstest]
    #[case((0, 0), -3.)]
    #[case((0, 1), 5.)]
    #[case((0, 2), 0.)]
    #[case((1, 0), 1.)]
    #[case((1, 1), -2.)]
    #[case((1, 2), -7.)]
    #[case((2, 0), 0.)]
    #[case((2, 1), 1.)]
    #[case((2, 2), 1.)]
    fn constructing_and_inspection_3x3(#[case] index: (usize, usize), #[case] expected: f32) {
        let m = Matrix3x3::new([-3., 5., 0.], [1., -2., -7.], [0., 1., 1.]);
        assert_eq!(m[index.0][index.1], expected);
    }

    #[test]
    fn comparing_4x4_same_are_equal() {
        let m4x4 = Matrix4x4::new(
            [1., 2., 3., 4.],
            [5.5, 6.5, 7.5, 8.5],
            [9., 10., 11., 12.],
            [13.5, 14.5, 15.5, 16.5],
        );
        let om4x4 = Matrix4x4::new(
            [1., 2., 3., 4.],
            [5.5, 6.5, 7.5, 8.5],
            [9., 10., 11., 12.],
            [13.5, 14.5, 15.5, 16.5],
        );
        assert_eq!(m4x4, om4x4);
    }

    #[test]
    fn comparing_4x4_diff_are_not_equal() {
        let m4x4 = Matrix4x4::new(
            [1., 2., 3., 4.2],
            [5.5, 6.5, 7.5, 8.5],
            [9., 10., 11., 12.],
            [13.5, 14.5, 15.5, 16.5],
        );
        let om4x4 = Matrix4x4::new(
            [1., 2., 3., 4.],
            [5.5, 6.5, 7.5, 8.5],
            [9., 10., 11., 12.],
            [13.5, 14.5, 15.5, 16.5],
        );
        assert_ne!(m4x4, om4x4);
    }

    #[test]
    fn comparing_3x3_same_are_equal() {
        let m3x3 = Matrix3x3::new([1., 2., 3.], [5.5, 6.5, 7.5], [9., 10., 11.]);
        let om3x3 = Matrix3x3::new([1., 2., 3.], [5.5, 6.5, 7.5], [9., 10., 11.]);
        assert_eq!(m3x3, om3x3);
    }

    #[test]
    fn comparing_3x3_diff_are_not_equal() {
        let m3x3 = Matrix3x3::new([1.1, 2., 3.], [5.5, 6.5, 7.5], [9., 10., 11.]);
        let om3x3 = Matrix3x3::new([1., 2., 3.], [5.5, 6.5, 7.5], [9., 10., 11.]);
        assert_ne!(m3x3, om3x3);
    }

    #[test]
    fn comparing_2x2_same_are_equal() {
        let m2x2 = Matrix2x2::new([1., 2.], [5.5, 6.5]);
        let om2x2 = Matrix2x2::new([1., 2.], [5.5, 6.5]);
        assert_eq!(m2x2, om2x2);
    }

    #[test]
    fn comparing_2x2_diff_are_not_equal() {
        let m2x2 = Matrix2x2::new([1.1, 2.], [5.5, 6.5]);
        let om2x2 = Matrix2x2::new([1., 2.], [5.5, 6.5]);
        assert_ne!(m2x2, om2x2);
    }
}

#[cfg(test)]
mod matrix_arithmetics {
    use crate::matrix::{Matrix2x2, Matrix3x3, Matrix4x4, Translation};
    use crate::point::Point;
    use crate::tuple::Tuple;
    use crate::vector::Vector;

    #[test]
    fn multiplication_other_matrix() {
        let m4x4 = Matrix4x4::new(
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 8., 7., 6.],
            [5., 4., 3., 2.],
        );
        let om4x4 = Matrix4x4::new(
            [-2., 1., 2., 3.],
            [3., 2., 1., -1.],
            [4., 3., 6., 5.],
            [1., 2., 7., 8.],
        );
        let expected = Matrix4x4::new(
            [20., 22., 50., 48.],
            [44., 54., 114., 108.],
            [40., 58., 110., 102.],
            [16., 26., 46., 42.],
        );

        assert_eq!(m4x4 * om4x4, expected);
    }
    #[test]
    fn multiplication_other_tuple() {
        let m = Matrix4x4::new(
            [1., 2., 3., 4.],
            [2., 4., 4., 2.],
            [8., 6., 4., 1.],
            [0., 0., 0., 1.],
        );
        let other = Tuple::new(1., 2., 3., 1.);

        assert_eq!(m * other, Tuple::new(18., 24., 33., 1.));
    }

    #[test]
    fn multiply_identity_matrix_by_matrix() {
        let m = Matrix4x4::new(
            [0., 1., 2., 4.],
            [1., 2., 4., 8.],
            [2., 4., 8., 16.],
            [4., 8., 16., 32.],
        );
        assert_eq!(m * Matrix4x4::identity(), m)
    }
    #[test]
    fn multiply_identity_matrix_by_tuple() {
        let tuple = Tuple::new(1., 2., 3., 4.);
        assert_eq!(Matrix4x4::identity() * tuple, tuple);
    }

    #[test]
    fn transposing_matrix() {
        let m = Matrix4x4::new(
            [0., 9., 3., 0.],
            [9., 8., 0., 8.],
            [1., 8., 5., 3.],
            [0., 0., 5., 8.],
        );
        let expected = Matrix4x4::new(
            [0., 9., 1., 0.],
            [9., 8., 8., 0.],
            [3., 0., 5., 5.],
            [0., 8., 3., 8.],
        );
        assert_eq!(m.transpose(), expected);
    }

    #[test]
    fn transposing_identity_matrix() {
        assert_eq!(Matrix4x4::identity().transpose(), Matrix4x4::identity());
    }

    #[test]
    fn determinants_2x2() {
        let m = Matrix2x2::new([1., 5.], [-3., 2.]);
        assert_eq!(m.determinant(), 17.)
    }

    #[test]
    fn sub_matrix_of_3x3() {
        let m = Matrix3x3::new([1., 5., 0.], [-3., 2., 7.], [0., 6., -3.]);
        assert_eq!(m.submatrix(0, 2), Matrix2x2::new([-3., 2.], [0., 6.]));
    }

    #[test]
    fn sub_matrix_of_4x4() {
        let m = Matrix4x4::new(
            [-6., 1., 1., 6.],
            [-8., 5., 8., 6.],
            [-1., 0., 8., 2.],
            [-7., 1., -1., 1.],
        );
        assert_eq!(
            m.submatrix(2, 1),
            Matrix3x3::new([-6., 1., 6.], [-8., 8., 6.], [-7., -1., 1.])
        );
    }

    #[test]
    fn minor_of_3x3() {
        let m = Matrix3x3::new([3., 5., 0.], [2., -1., -7.], [6., -1., 5.]);
        assert_eq!(m.minor(1, 0), 25.);
    }

    #[test]
    fn cofactor_of_3x3() {
        let m = Matrix3x3::new([3., 5., 0.], [2., -1., -7.], [6., -1., 5.]);
        assert_eq!(m.cofactor(0, 0), -12.);
        assert_eq!(m.cofactor(1, 0), -25.);
    }

    #[test]
    fn determinant_of_3x3() {
        let m = Matrix3x3::new([1., 2., 6.], [-5., 8., -4.], [2., 6., 4.]);
        assert_eq!(m.cofactor(0, 0), 56.);
        assert_eq!(m.cofactor(0, 1), 12.);
        assert_eq!(m.cofactor(0, 2), -46.);
        assert_eq!(m.determinant(), -196.);
    }

    #[test]
    fn determinant_of_4x4() {
        let m = Matrix4x4::new(
            [-2., -8., 3., 5.],
            [-3., 1., 7., 3.],
            [1., 2., -9., 6.],
            [-6., 7., 7., -9.],
        );
        assert_eq!(m.cofactor(0, 0), 690.);
        assert_eq!(m.cofactor(0, 1), 447.);
        assert_eq!(m.cofactor(0, 2), 210.);
        assert_eq!(m.cofactor(0, 3), 51.);
        assert_eq!(m.determinant(), -4071.);
    }

    #[test]
    fn non_invertible_matrix() {
        let m = Matrix4x4::new(
            [6., 4., 4., 4.],
            [5., 5., 7., 6.],
            [4., -9., 3., -7.],
            [9., 1., 7., -6.],
        );
        assert_eq!(m.determinant(), -2120.);
        assert!(m.is_invertible());
    }
    #[test]
    fn invertible_matrix() {
        let m = Matrix4x4::new(
            [-4., 2., -2., -3.],
            [9., 6., 2., 6.],
            [0., -5., 1., -5.],
            [0., 0., 0., 0.],
        );
        assert_eq!(m.determinant(), 0.);
        assert!(!m.is_invertible());
    }

    #[test]
    fn inverse_of_matrix() {
        let m = Matrix4x4::new(
            [-5., 2., 6., -8.],
            [1., -5., 1., 8.],
            [7., 7., -6., -7.],
            [1., -3., 7., 4.],
        );
        let b = m.inverse().unwrap();
        assert_eq!(m.determinant(), 532.);
        assert_eq!(m.cofactor(2, 3), -160.);
        assert_eq!(b[3][2], -160. / 532.);
        assert_eq!(m.cofactor(3, 2), 105.);
        assert_eq!(b[2][3], 105. / 532.);
        assert_eq!(
            b,
            Matrix4x4::new(
                [0.21805, 0.45113, 0.24060, -0.04511],
                [-0.80827, -1.45677, -0.44361, 0.52068],
                [-0.07895, -0.22368, -0.05263, 0.19737],
                [-0.52256, -0.81391, -0.30075, 0.30639]
            )
        );
    }

    #[test]
    fn testing_inverse_matrix() {
        let m = Matrix4x4::new(
            [8., -5., 9., 2.],
            [7., 5., 6., 1.],
            [-6., 0., 9., 6.],
            [-3., 0., -9., -4.],
        );
        let expected = Matrix4x4::new(
            [-0.15385, -0.15385, -0.28205, -0.53846],
            [-0.07692, 0.12308, 0.02564, 0.03077],
            [0.35897, 0.35897, 0.43590, 0.92308],
            [-0.69231, -0.69231, -0.76923, -1.92308],
        );
        assert_eq!(m.inverse(), Some(expected));
        let m = Matrix4x4::new(
            [9., 3., 0., 9.],
            [-5., -2., -6., -3.],
            [-4., 9., 6., 4.],
            [-7., 6., 6., 2.],
        );
        let expected = Matrix4x4::new(
            [-0.04074, -0.07778, 0.14444, -0.22222],
            [-0.07778, 0.03333, 0.36667, -0.33333],
            [-0.02901, -0.14630, -0.10926, 0.12963],
            [0.17778, 0.06667, -0.26667, 0.33333],
        );
        assert_eq!(m.inverse(), Some(expected));
    }

    #[test]
    fn multiplying_a_product_by_its_inverse() {
        let a = Matrix4x4::new(
            [3., -9., 7., 3.],
            [3., -8., 2., -9.],
            [-4., 4., 4., 1.],
            [-6., 5., -1., 1.],
        );
        let b = Matrix4x4::new(
            [8., 2., 2., 2.],
            [3., -1., 7., 0.],
            [7., 0., 5., 4.],
            [6., -2., 0., 5.],
        );
        let c = a * b;
        assert_eq!(c * b.inverse().unwrap(), a);
    }

    #[test]
    fn translation_matrix_structure() {
        let transform = Translation::new(5., -3., 2.);
        assert_eq!(
            transform,
            Matrix4x4::new(
                [1., 0., 0., 5.],
                [0., 1., 0., -3.],
                [0., 0., 1., 2.],
                [0., 0., 0., 1.]
            )
        );
    }

    #[test]
    fn multiplying_a_translation_matrix() {
        let transform = Translation::new(5., -3., 2.);
        let point = Point::new(-3., 4., 5.);
        assert_eq!(transform * point, Point::new(2., 1., 7.));
    }

    #[test]
    fn multiplying_by_inverse_of_translation() {
        let transform = Translation::new(5., -3., 2.);
        let inverse = transform.inverse().unwrap();
        let point = Point::new(-3., 4., 5.);
        assert_eq!(inverse * point, Point::new(-8., 7., 3.));
    }

    #[test]
    fn translation_does_not_affect_vectors() {
        let transform = Translation::new(5., -3., 2.);
        let v = Vector::new(-3., 4., 5.);
        assert_eq!(transform * v, v);
    }
}

#[cfg(test)]
mod scaling_tests {
    use crate::matrix::Scaling;
    use crate::point::Point;
    use crate::vector::Vector;

    #[test]
    fn scaling_matrix_applied_to_point() {
        let scaling = Scaling::new(2., 3., 4.);
        let point = Point::new(-4., 6., 8.);
        assert_eq!(scaling * point, Point::new(-8., 18., 32.));
    }

    #[test]
    fn scaling_matrix_applied_to_vector() {
        let scaling = Scaling::new(2., 3., 4.);
        let vector = Vector::new(-4., 6., 8.);
        assert_eq!(scaling * vector, Vector::new(-8., 18., 32.));
    }

    #[test]
    fn scaling_inverse_matrix_applied_to_vector() {
        let scaling = Scaling::new(2., 3., 4.);
        let inverse_scaling = scaling.inverse().unwrap();
        let vector = Vector::new(-4., 6., 8.);
        assert_eq!(inverse_scaling * vector, Vector::new(-2., 2., 2.));
    }

    #[test]
    fn reflection_is_scaling_by_negative_value() {
        let transform = Scaling::new(-1., 1., 1.);
        let point = Point::new(2., 3., 4.);
        assert_eq!(transform * point, Point::new(-2., 3., 4.));
    }
}
