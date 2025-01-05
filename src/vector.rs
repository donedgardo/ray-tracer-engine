use crate::coord::Coord;
use crate::float_eq::f32_are_eq;
use crate::tuple::Tuple;
use std::ops::{Mul, Neg, Sub};

#[derive(Debug)]
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

impl Mul for Vector {
    type Output = f32;

    fn mul(self, rhs: Self) -> Self::Output {
        self.0 * rhs.0
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

#[cfg(test)]
mod vector_tests {
    use crate::tuple::Tuple;
    use crate::vector::Vector;

    #[test]
    fn new_vector_creates_tuple_with_w_0() {
        let v = Vector::new(4., -4., 3.);
        assert_eq!(v, Tuple::new(4., -4., 3., 0.));
    }
}
