use crate::coord::Coord;
use crate::float_eq::f32_are_eq;
use crate::tuple::Tuple;
use crate::vector::Vector;
use std::ops::{Add, Sub};

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

impl Add<Vector> for Point {
    type Output = Point;

    fn add(self, rhs: Vector) -> Self::Output {
        (self.0 + rhs.into()).into()
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

impl From<Tuple> for Point {
    fn from(tuple: Tuple) -> Self {
        Point::new(tuple.x(), tuple.y(), tuple.z())
    }
}

impl From<Point> for Tuple {
    fn from(value: Point) -> Self {
        Tuple::new(value.x(), value.y(), value.z(), 1.)
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

    #[test]
    fn can_create_from_tuple() {
        let p: Point = Tuple::new(1., 2., 3., 4.).into();
        assert_eq!(p, Point::new(1., 2., 3.));
    }

    #[test]
    fn can_create_tuple_from_point() {
        let p: Tuple = Point::new(1., 2., 3.).into();
        assert_eq!(p, Tuple::new(1., 2., 3., 1.));
    }
}
