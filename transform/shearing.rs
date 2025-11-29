use crate::Shearing;

impl Shearing {
    pub fn new(xy: f32, xz: f32, yx: f32, yz: f32, zx: f32, zy: f32) -> Matrix4x4 {
        Matrix4x4::new(
            [1., xy, xz, 0.],
            [yx, 1., yz, 0.],
            [zx, zy, 1., 0.],
            [0., 0., 0., 1.],
        )
    }
}

#[cfg(test)]
mod shearing_tests {
    use crate::point::Point;
    use crate::transform::Shearing;
    use rstest::rstest;

    #[test]
    fn moves_x_in_proportion_to_y() {
        let transform = Shearing::new(1., 0., 0., 0., 0., 0.);
        let point = Point::new(2., 3., 4.);
        assert_eq!(Point::new(5., 3., 4.), point * transform);
    }

    #[test]
    fn moves_x_in_proportion_to_z() {
        let transform = Shearing::new(0., 1., 0., 0., 0., 0.);
        let point = Point::new(2., 3., 4.);
        assert_eq!(Point::new(6., 3., 4.), point * transform);
    }

    #[rstest]
    #[case((1., 0., 0., 0., 0., 0.), (5., 3., 4. ))]
    #[case((0., 1., 0., 0., 0., 0.), (6., 3., 4. ))]
    #[case((0., 0., 1., 0., 0., 0.), (2., 5., 4. ))]
    #[case((0., 0., 0., 1., 0., 0.), (2., 7., 4. ))]
    #[case((0., 0., 0., 0., 1., 0.), (2., 3., 6. ))]
    #[case((0., 0., 0., 0., 0., 1.), (2., 3., 7. ))]
    fn moves_in_proportion(
        #[case] shear: (f32, f32, f32, f32, f32, f32),
        #[case] expected_pos: (f32, f32, f32),
    ) {
        let transform = Shearing::new(shear.0, shear.1, shear.2, shear.3, shear.4, shear.5);
        let point = Point::new(2., 3., 4.);
        assert_eq!(
            Point::new(expected_pos.0, expected_pos.1, expected_pos.2),
            point * transform
        );
    }
}
