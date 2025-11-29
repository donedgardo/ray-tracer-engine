use crate::transforms::matrix::Matrix4x4;

pub(crate) struct Translation;

impl Translation {
    pub fn new(x: f32, y: f32, z: f32) -> Matrix4x4 {
        Matrix4x4::new(
            [1., 0., 0., x],
            [0., 1., 0., y],
            [0., 0., 1., z],
            [0., 0., 0., 1.],
        )
    }
}

#[cfg(test)]
mod translation_tests {
    use crate::point::Point;
    use crate::transforms::matrix::Matrix4x4;
    use crate::transforms::translation::Translation;
    use crate::vector::Vector;

    #[test]
    fn translation_matrix_structure() {
        let transform = Translation::new(5., -3., 2.);
        assert_eq!(
            transform,
            Matrix4x4::new(
                [1., 0., 0., 5.],
                [0., 1., 0., -3.],
                [0., 0., 1., 2.],
                [0., 0., 0., 1.]
            )
        );
    }

    #[test]
    fn multiplying_a_translation_matrix() {
        let transform = Translation::new(5., -3., 2.);
        let point = Point::new(-3., 4., 5.);
        assert_eq!(transform * point, Point::new(2., 1., 7.));
    }

    #[test]
    fn multiplying_by_inverse_of_translation() {
        let transform = Translation::new(5., -3., 2.);
        let inverse = transform.inverse().unwrap();
        let point = Point::new(-3., 4., 5.);
        assert_eq!(inverse * point, Point::new(-8., 7., 3.));
    }

    #[test]
    fn translation_does_not_affect_vectors() {
        let transform = Translation::new(5., -3., 2.);
        let v = Vector::new(-3., 4., 5.);
        assert_eq!(transform * v, v);
    }
}
