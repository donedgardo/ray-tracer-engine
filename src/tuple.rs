use crate::coord::Coord;
use crate::float_eq::f32_are_eq;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug)]
pub struct Tuple(f32, f32, f32, f32);

impl Tuple {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(x, y, z, w)
    }

    pub fn w(&self) -> f32 {
        self.3
    }

    pub fn is_point(&self) -> bool {
        self.3 == 1.
    }

    pub fn is_vector(&self) -> bool {
        self.3 == 0.
    }

    pub fn magnitude(&self) -> f32 {
        (self.x() * self.x() + self.y() * self.y() + self.z() * self.z() + self.w() * self.w())
            .sqrt()
    }

    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if f32_are_eq(mag, 0.0) {
            return Self::new(self.x(), self.y(), self.z(), self.w());
        }
        Self::new(
            self.x() / mag,
            self.y() / mag,
            self.z() / mag,
            self.w() / mag,
        )
    }
}

impl Coord for Tuple {
    fn x(&self) -> f32 {
        self.0
    }

    fn y(&self) -> f32 {
        self.1
    }

    fn z(&self) -> f32 {
        self.2
    }
}

impl Add for Tuple {
    type Output = Tuple;

    fn add(self, rhs: Self) -> Self::Output {
        Tuple::new(
            self.x() + rhs.x(),
            self.y() + rhs.y(),
            self.z() + rhs.z(),
            self.w() + rhs.w(),
        )
    }
}

impl Sub for Tuple {
    type Output = Tuple;

    fn sub(self, rhs: Self) -> Self::Output {
        Tuple::new(
            self.x() - rhs.x(),
            self.y() - rhs.y(),
            self.z() - rhs.z(),
            self.w() - rhs.w(),
        )
    }
}

impl Mul<f32> for Tuple {
    type Output = Tuple;

    fn mul(self, rhs: f32) -> Self::Output {
        Tuple::new(
            self.x() * rhs,
            self.y() * rhs,
            self.z() * rhs,
            self.w() * rhs,
        )
    }
}

impl Mul for Tuple {
    type Output = f32;

    fn mul(self, rhs: Tuple) -> Self::Output {
        self.x() * rhs.x() + self.y() * rhs.y() + self.z() * rhs.z() + self.w() * rhs.w()
    }
}

impl Div<f32> for Tuple {
    type Output = Tuple;

    fn div(self, rhs: f32) -> Self::Output {
        Tuple::new(
            self.x() / rhs,
            self.y() / rhs,
            self.z() / rhs,
            self.w() / rhs,
        )
    }
}

impl PartialEq for Tuple {
    fn eq(&self, other: &Self) -> bool {
        f32_are_eq(self.x(), other.x())
            && f32_are_eq(self.y(), other.y())
            && f32_are_eq(self.z(), other.z())
            && (self.w() == other.w())
    }
}

impl Neg for Tuple {
    type Output = Tuple;

    fn neg(self) -> Self::Output {
        Tuple::new(-self.x(), -self.y(), -self.z(), -self.w())
    }
}

#[cfg(test)]
mod point_tuple_tests {
    use crate::coord::Coord;
    use crate::tuple::Tuple;

    #[test]
    fn has_x() {
        let p = create_point_tuple();
        assert_eq!(p.x(), 4.3);
    }

    #[test]
    fn has_y() {
        let p = create_point_tuple();
        assert_eq!(p.y(), -4.2);
    }

    #[test]
    fn has_z() {
        let p = create_point_tuple();
        assert_eq!(p.z(), 3.1);
    }

    #[test]
    fn is_point() {
        let p = create_point_tuple();
        assert!(p.is_point());
    }

    fn create_point_tuple() -> Tuple {
        Tuple::new(4.3, -4.2, 3.1, 1.)
    }
}

#[cfg(test)]
mod vector_tuple_tests {
    use crate::coord::Coord;
    use crate::tuple::Tuple;

    #[test]
    fn has_x() {
        let v = create_vector_tuple();
        assert_eq!(v.x(), 4.3);
    }

    #[test]
    fn has_y() {
        let v = create_vector_tuple();
        assert_eq!(v.y(), -4.2);
    }

    #[test]
    fn has_z() {
        let v = create_vector_tuple();
        assert_eq!(v.z(), 3.1);
    }

    #[test]
    fn is_not_point() {
        let v = create_vector_tuple();
        assert!(!v.is_point());
    }

    #[test]
    fn is_vector() {
        let v = create_vector_tuple();
        assert!(v.is_vector());
    }

    fn create_vector_tuple() -> Tuple {
        Tuple::new(4.3, -4.2, 3.1, 0.)
    }
}

#[cfg(test)]
mod tuple_equality_tests {
    use crate::tuple::Tuple;

    #[test]
    fn when_close_to_e_minus_5_are_equal() {
        let a = Tuple(0.00001, 0.2, 0.3, 0.4);
        let b = Tuple(0.000009, 0.2, 0.3, 0.4);
        assert_eq!(a, b);
    }

    #[test]
    fn when_diff_bigger_than_e_minus_5_not_equal() {
        let a = Tuple(0.00001, 0.2, 0.3, 0.4);
        let b = Tuple(0.00002, 0.2, 0.3, 0.4);
        assert_ne!(a, b);
    }
}

#[cfg(test)]
mod tuple_arithmetic_tests {
    use crate::float_eq::f32_are_eq;
    use crate::point::Point;
    use crate::tuple::Tuple;
    use crate::vector::Vector;
    use rstest::rstest;
    use std::ops::Mul;

    #[test]
    fn tuples_can_be_added() {
        let a = Tuple::new(3., -2., 5., 1.);
        let b = Tuple::new(-2., 3., 1., 0.);
        let c = a + b;
        assert_eq!(c, Tuple::new(1., 1., 6., 1.));
    }

    #[test]
    fn it_can_subtract_two_points() {
        let a = Point::new(3., 2., 1.);
        let b = Point::new(5., 6., 7.);
        let c = a - b;
        assert_eq!(c, Vector::new(-2., -4., -6.));
    }

    #[test]
    fn subtracting_a_vector_from_a_point() {
        let p = Point::new(3., 2., 1.);
        let v = Vector::new(5., 6., 7.);
        let c = p - v;
        assert_eq!(c, Point::new(-2., -4., -6.));
    }

    #[test]
    fn subtracting_two_vectors() {
        let a = Vector::new(3., 2., 1.);
        let b = Vector::new(5., 6., 7.);
        let c = a - b;
        assert_eq!(c, Vector::new(-2., -4., -6.));
    }

    #[test]
    fn subtracting_a_vector_from_the_zero_vector() {
        let zero = Vector::new(0., 0., 0.);
        let v = Vector::new(1., -2., 3.);
        assert_eq!(zero - v, Vector::new(-1., 2., -3.));
    }

    #[test]
    fn negating_a_tuple() {
        let v = Tuple::new(1., -2., 3., -4.);
        assert_eq!(-v, Tuple::new(-1., 2., -3., 4.));
    }

    #[test]
    fn negating_a_vector() {
        let v = Vector::new(1., -2., 3.);
        assert_eq!(-v, Vector::new(-1., 2., -3.));
    }

    #[test]
    fn multiplying_tuple_by_scalar() {
        let a = Tuple::new(1., -2., 3., -4.);
        let c = a * 3.5;
        assert_eq!(c, Tuple::new(3.5, -7., 10.5, -14.));
    }

    #[test]
    fn multiplying_tuple_by_fraction() {
        let a = Tuple::new(1., -2., 3., -4.);
        let c = a * 0.5;
        assert_eq!(c, Tuple::new(0.5, -1., 1.5, -2.));
    }

    #[test]
    fn dividing_tuple_by_scalar() {
        let a = Tuple::new(1., -2., 3., -4.);
        let c = a / 2.;
        assert_eq!(c, Tuple::new(0.5, -1., 1.5, -2.));
    }

    #[rstest]
    #[case(Vector::new(1., 0., 0.), 1.)]
    #[case(Vector::new(0., 1., 0.), 1.)]
    #[case(Vector::new(1., 2., 3.), 14.0f32.sqrt())]
    #[case(Vector::new(-1., -2., -3.), 14.0f32.sqrt())]
    fn computing_magnitude_of_vector(#[case] v: Vector, #[case] expected: f32) {
        assert!(f32_are_eq(v.magnitude(), expected));
    }

    #[rstest]
    #[case(Vector::new(4., 0., 0.), Vector::new(1., 0., 0.))]
    #[case(Vector::new(1., 2., 3.),
        Vector::new(1. / 14f32.sqrt(), 2. / 14f32.sqrt(), 3. / 14f32.sqrt()))]
    fn normalizing_vector(#[case] v: Vector, #[case] expected: Vector) {
        assert_eq!(v.normalize(), expected);
    }

    #[test]
    fn mag_of_normalized_vector_is_1() {
        let v = Vector::new(1., 2., 3.);
        let normalized = v.normalize();
        assert!(f32_are_eq(normalized.magnitude(), 1.));
    }

    #[test]
    fn dot_product() {
        let a = Vector::new(1., 2., 3.);
        let b = Vector::new(2., 3., 4.);
        assert!(f32_are_eq(a * b, 20.))
    }

    #[test]
    fn cross_product() {
        let a = Vector::new(1., 2., 3.);
        let b = Vector::new(2., 3., 4.);
        assert_eq!(a.cross(&b), Vector::new(-1., 2., -1.));
        assert_eq!(b.cross(&a), Vector::new(1., -2., 1.));
    }
}
