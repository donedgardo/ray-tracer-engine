use crate::transforms::matrix::Matrix4x4;
use crate::transforms::rotation::Rotation;
use crate::transforms::scaling::Scaling;
use translation::Translation;

pub mod matrix;
pub mod rotation;
pub mod scaling;
pub mod shearing;
mod translation;

pub struct Transform;
impl Transform {
    pub fn identity() -> Matrix4x4 {
        Matrix4x4::identity()
    }
}

pub trait Transformable {
    fn rotate_x(self, radians: f32) -> Self;
    fn rotate_y(self, radians: f32) -> Self;
    fn rotate_z(self, radians: f32) -> Self;
    fn scale(self, x: f32, y: f32, z: f32) -> Self;
    fn translate(self, x: f32, y: f32, z: f32) -> Self;
}

impl Transformable for Matrix4x4 {
    fn rotate_x(self, radians: f32) -> Self {
        Rotation::x(radians) * self
    }

    fn rotate_y(self, radians: f32) -> Self {
        Rotation::y(radians) * self
    }

    fn rotate_z(self, radians: f32) -> Self {
        Rotation::z(radians) * self
    }

    fn scale(self, x: f32, y: f32, z: f32) -> Self {
        Scaling::new(x, y, z) * self
    }

    fn translate(self, x: f32, y: f32, z: f32) -> Self {
        Translation::new(x, y, z) * self
    }
}
#[cfg(test)]
mod transformation_tests {
    use crate::point::Point;
    use crate::transforms::rotation::Rotation;
    use crate::transforms::scaling::Scaling;
    use crate::transforms::translation::Translation;
    use crate::transforms::{Transform, Transformable};
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn individual_transforms_in_sequence() {
        let p = Point::new(1., 0., 1.);
        let a = Rotation::x(FRAC_PI_2);
        let b = Scaling::new(5., 5., 5.);
        let c = Translation::new(10., 5., 7.);

        let p2 = a * p;
        assert_eq!(p2, Point::new(1., -1., 0.));
        let p3 = b * p2;
        assert_eq!(p3, Point::new(5., -5., 0.));
        let p4 = c * p3;
        assert_eq!(p4, Point::new(15., 0., 7.));
    }

    #[test]
    fn chain_transforms_applied_in_reverse() {
        let p = Point::new(1., 0., 1.);
        let a = Rotation::x(FRAC_PI_2);
        let b = Scaling::new(5., 5., 5.);
        let c = Translation::new(10., 5., 7.);

        let t = c * b * a;
        assert_eq!(t * p, Point::new(15., 0., 7.));
    }

    #[test]
    fn flexible_api() {
        let p = Point::new(1., 0., 1.);
        let transform = Transform::identity()
            .rotate_x(FRAC_PI_2)
            .scale(5., 5., 5.)
            .translate(10., 5., 7.);
        assert_eq!(transform * p, Point::new(15., 0., 7.));
    }
}
