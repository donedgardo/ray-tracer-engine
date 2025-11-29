use crate::matrix::Matrix4x4;
use crate::point::Point;
use crate::tuple::Tuple;
use crate::vector::Vector;
use std::ops::Mul;

pub struct Shearing;

pub struct Scaling(Matrix4x4);
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

#[cfg(test)]
mod scaling_tests {
    use crate::point::Point;
    use crate::transform::Scaling;
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
