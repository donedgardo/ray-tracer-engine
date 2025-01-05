use crate::coord::Coord;
use crate::float_eq::f32_are_eq;
use crate::tuple::Tuple;
use crate::vector::Vector;
use std::ops::Sub;

#[derive(Debug)]
pub struct Point(Tuple);

impl Point {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(Tuple::new(x, y, z, 1.))
    }
}

impl Coord for Point {
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

impl Sub for Point {
    type Output = Vector;

    fn sub(self, rhs: Self) -> Self::Output {
        (self.0 - rhs.0).into()
    }
}

impl Sub<Vector> for Point {
    type Output = Point;

    fn sub(self, rhs: Vector) -> Self::Output {
        Point::new(self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z())
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Point) -> bool {
        f32_are_eq(self.x(), other.x())
            && f32_are_eq(self.0.y(), other.y())
            && f32_are_eq(self.0.z(), other.z())
    }
}

impl PartialEq<Tuple> for Point {
    fn eq(&self, other: &Tuple) -> bool {
        f32_are_eq(self.x(), other.x())
            && f32_are_eq(self.0.y(), other.y())
            && f32_are_eq(self.0.z(), other.z())
            && other.is_point()
    }
}

#[cfg(test)]
mod point_tests {
    use crate::point::Point;
    use crate::tuple::Tuple;

    #[test]
    fn new_point_equals_tuple_with_w_1() {
        let p = Point::new(4., -4., 3.);
        assert_eq!(p, Tuple::new(4., -4., 3., 1.));
    }
}
