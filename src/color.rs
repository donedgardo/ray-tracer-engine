use crate::coord::Coord;
use crate::tuple::Tuple;
use std::ops::{Add, Mul, Sub};

#[derive(Debug, PartialEq, Clone)]
pub struct Color(Tuple);

impl Color {
    pub fn new(red: f32, green: f32, blue: f32) -> Self {
        Self(Tuple::new(red, green, blue, 1.))
    }
    pub fn red(&self) -> f32 {
        self.0.x()
    }

    pub fn green(&self) -> f32 {
        self.0.y()
    }

    pub fn blue(&self) -> f32 {
        self.0.z()
    }
}

impl Add for Color {
    type Output = Color;

    fn add(self, rhs: Self) -> Self::Output {
        (self.0 + rhs.0).into()
    }
}

impl Sub for Color {
    type Output = Color;

    fn sub(self, rhs: Self) -> Self::Output {
        (self.0 - rhs.0).into()
    }
}

impl Mul<f32> for Color {
    type Output = Color;

    fn mul(self, rhs: f32) -> Self::Output {
        (self.0 * rhs).into()
    }
}

impl Mul for Color {
    type Output = Color;

    fn mul(self, rhs: Color) -> Self::Output {
        Color::new(
            self.red() * rhs.red(),
            self.green() * rhs.green(),
            self.blue() * rhs.blue(),
        )
    }
}

impl From<Tuple> for Color {
    fn from(t: Tuple) -> Self {
        Color::new(t.x(), t.y(), t.z())
    }
}

#[cfg(test)]
mod color_tests {
    use crate::color::Color;

    #[test]
    fn has_red_channel() {
        let color = create_color();
        assert_eq!(color.red(), -0.5);
    }

    #[test]
    fn has_blue_channel() {
        let color = create_color();
        assert_eq!(color.blue(), 1.7);
    }

    #[test]
    fn has_green_channel() {
        let color = create_color();
        assert_eq!(color.green(), 0.4);
    }

    #[test]
    fn adding_colors() {
        let a = Color::new(0.9, 0.6, 0.75);
        let b = Color::new(0.7, 0.1, 0.25);
        assert_eq!(a + b, Color::new(1.6, 0.7, 1.0));
    }

    #[test]
    fn subtracting_colors() {
        let a = Color::new(0.9, 0.6, 0.75);
        let b = Color::new(0.7, 0.1, 0.25);
        assert_eq!(a - b, Color::new(0.2, 0.5, 0.5));
    }

    #[test]
    fn multiplying_by_scalar() {
        let c = Color::new(0.2, 0.3, 0.4);
        assert_eq!(c * 2., Color::new(0.4, 0.6, 0.8));
    }

    #[test]
    fn multiplying_two_colors() {
        let a = Color::new(1., 0.2, 0.4);
        let b = Color::new(0.9, 1.0, 0.1);

        assert_eq!(a * b, Color::new(0.9, 0.2, 0.04));
    }

    fn create_color() -> Color {
        let color = Color::new(-0.5, 0.4, 1.7);
        color
    }
}
