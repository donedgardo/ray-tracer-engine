use crate::matrix::Matrix4x4;
struct Rotation;
impl Rotation {
    pub fn x(radians: f32) -> Matrix4x4 {
        Matrix4x4::new(
            [1., 0., 0., 0.],
            [0., radians.cos(), -radians.sin(), 0.],
            [0., radians.sin(), radians.cos(), 0.],
            [0., 0., 0., 1.],
        )
    }
    pub fn y(radians: f32) -> Matrix4x4 {
        Matrix4x4::new(
            [radians.cos(), 0., radians.sin(), 0.],
            [0., 1., 0., 0.],
            [-radians.sin(), 0., radians.cos(), 0.],
            [0., 0., 0., 1.],
        )
    }
    pub fn z(radians: f32) -> Matrix4x4 {
        Matrix4x4::new(
            [radians.cos(), -radians.sin(), 0., 0.],
            [radians.sin(), radians.cos(), 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        )
    }
}

pub fn degree_to_radians(degrees: f32) -> f32 {
    degrees * std::f32::consts::PI / 180.
}

#[cfg(test)]
mod rotation_tests {
    use crate::point::Point;
    use crate::rotation::Rotation;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    #[test]
    fn rotating_around_x_axis_half_quarter() {
        let p = Point::new(0., 1., 0.);
        let half_quarter = Rotation::x(FRAC_PI_4);
        let y_z: f32 = 2_f32.sqrt() / 2.;
        assert_eq!(half_quarter * p, Point::new(0., y_z, y_z));
    }

    #[test]
    fn rotating_around_x_axis_full_quarter() {
        let p = Point::new(0., 1., 0.);
        let full_quarter = Rotation::x(FRAC_PI_2);
        assert_eq!(full_quarter * p, Point::new(0., 0., 1.));
    }

    #[test]
    fn inverse_of_x_rotation_rotates_opposite_direction() {
        let p = Point::new(0., 1., 0.);
        let half_quarter = Rotation::x(FRAC_PI_4);
        let inversion = half_quarter.inverse().unwrap();
        let y_z: f32 = 2_f32.sqrt() / 2.;
        assert_eq!(inversion * p, Point::new(0., y_z, -y_z));
    }
    #[test]
    fn inverse_rotating_around_x_axis_full_quarter_opposite_direction() {
        let p = Point::new(0., 1., 0.);
        let full_quarter = Rotation::x(FRAC_PI_2);
        let inverse = full_quarter.inverse().unwrap();
        assert_eq!(inverse * p, Point::new(0., 0., -1.));
    }

    #[test]
    fn rotating_around_y_axis_half_quarter() {
        let p = Point::new(0., 0., 1.);
        let half_quarter = Rotation::y(FRAC_PI_4);
        let x_z: f32 = 2_f32.sqrt() / 2.;
        assert_eq!(half_quarter * p, Point::new(x_z, 0., x_z));
    }

    #[test]
    fn rotating_around_y_axis_full_quarter() {
        let p = Point::new(0., 0., 1.);
        let full_quarter = Rotation::y(FRAC_PI_2);
        assert_eq!(full_quarter * p, Point::new(1., 0., 0.));
    }

    #[test]
    fn rotating_around_z_axis_half_quarter() {
        let p = Point::new(0., 1., 0.);
        let half_quarter = Rotation::z(FRAC_PI_4);
        let x_y: f32 = 2_f32.sqrt() / 2.;
        assert_eq!(half_quarter * p, Point::new(-x_y, x_y, 0.));
    }

    #[test]
    fn rotating_around_z_axis_full_quarter() {
        let p = Point::new(0., 1., 0.);
        let full_quarter = Rotation::z(FRAC_PI_2);
        assert_eq!(full_quarter * p, Point::new(-1., 0., 0.));
    }
}

#[cfg(test)]
mod degree_to_radians_test {
    use crate::rotation::degree_to_radians;
    use std::f32::consts::PI;

    #[test]
    fn zero_test() {
        let radians = degree_to_radians(0.);
        assert_eq!(radians, 0.);
    }

    #[test]
    fn three_sixty_test() {
        let radians = degree_to_radians(360.);
        assert_eq!(radians, PI * 2.);
    }

    #[test]
    fn one_eighty_test() {
        let radians = degree_to_radians(180.);
        assert_eq!(radians, PI);
    }

    #[test]
    fn negative_one_eighty_test() {
        let radians = degree_to_radians(-180.);
        assert_eq!(radians, -PI);
    }
}
