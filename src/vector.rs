use crate::coord::Coord;
use crate::float_eq::f32_are_eq;
use crate::point::Point;
use crate::tuple::Tuple;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy)]
pub struct Vector(Tuple);

impl Vector {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(Tuple::new(x, y, z, 0.))
    }

    pub fn magnitude(&self) -> f32 {
        self.0.magnitude()
    }

    pub fn normalize(&self) -> Self {
        Self(self.0.normalize())
    }

    pub fn dot(&self, rhs: &Vector) -> f32 {
        self.0.dot(&rhs.0)
    }

    pub fn cross(&self, rhs: &Vector) -> Self {
        Self::new(
            self.y() * rhs.z() - self.z() * rhs.y(),
            self.z() * rhs.x() - self.x() * rhs.z(),
            self.x() * rhs.y() - self.y() * rhs.x(),
        )
    }
}

impl Coord for Vector {
    fn x(&self) -> f32 {
        self.0.x()
    }

    fn y(&self) -> f32 {
        self.0.y()
    }

    fn z(&self) -> f32 {
        self.0.z()
    }
}

impl PartialEq<Tuple> for Vector {
    fn eq(&self, other: &Tuple) -> bool {
        f32_are_eq(self.0.x(), other.x())
            && f32_are_eq(self.y(), other.y())
            && f32_are_eq(self.z(), other.z())
            && other.is_vector()
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Vector) -> bool {
        f32_are_eq(self.0.x(), other.x())
            && f32_are_eq(self.y(), other.y())
            && f32_are_eq(self.z(), other.z())
    }
}

impl Sub for Vector {
    type Output = Vector;

    fn sub(self, rhs: Self) -> Self::Output {
        Vector(self.0 - rhs.0)
    }
}

impl Add for Vector {
    type Output = Vector;

    fn add(self, rhs: Vector) -> Self::Output {
        (self.0 + rhs.0).into()
    }
}
impl Add<Point> for Vector {
    type Output = Point;

    fn add(self, rhs: Point) -> Self::Output {
        (self.0 + rhs.into()).into()
    }
}

impl Mul<f32> for Vector {
    type Output = Vector;

    fn mul(self, rhs: f32) -> Self::Output {
        (self.0 * rhs).into()
    }
}
impl Neg for Vector {
    type Output = Vector;

    fn neg(self) -> Self::Output {
        Vector(-self.0)
    }
}

impl From<Tuple> for Vector {
    fn from(value: Tuple) -> Self {
        Vector::new(value.x(), value.y(), value.z())
    }
}

impl From<Vector> for Tuple {
    fn from(value: Vector) -> Self {
        Tuple::new(value.x(), value.y(), value.z(), 0.0)
    }
}

#[cfg(test)]
mod vector_tests {
    use crate::tuple::Tuple;
    use crate::vector::Vector;

    #[test]
    fn new_vector_creates_tuple_with_w_0() {
        let v = Vector::new(4., -4., 3.);
        assert_eq!(v, Tuple::new(4., -4., 3., 0.));
    }

    #[test]
    fn can_turn_into_tuple() {
        let t: Tuple = Vector::new(4., -4., 3.).into();
        assert_eq!(t, Tuple::new(4., -4., 3., 0.));
    }
}
