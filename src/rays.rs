use crate::point::Point;
use crate::transforms::matrix::Matrix4x4;
use crate::vector::Vector;

pub struct Ray {
    origin: Point,
    direction: Vector,
}

impl Ray {
    pub fn new(origin: Point, direction: Vector) -> Self {
        Self { origin, direction }
    }
    pub fn origin(&self) -> &Point {
        &self.origin
    }
    pub fn direction(&self) -> &Vector {
        &self.direction
    }
    pub fn position(&self, time: f32) -> Point {
        self.origin.clone() + self.direction * time
    }
    pub fn transform(&self, transform: Matrix4x4) -> Self {
        Self {
            origin: transform * self.origin.clone(),
            direction: transform * self.direction.clone(),
        }
    }
}

#[cfg(test)]
mod rays_tests {
    use crate::point::Point;
    use crate::rays::Ray;
    use crate::transforms::{Transform, Transformable};
    use crate::vector::Vector;
    #[test]
    fn creating_and_querying() {
        let origin = Point::new(1., 2., 3.);
        let direction = Vector::new(4., 5., 6.);
        let ray = Ray::new(origin.clone(), direction.clone());
        assert_eq!(ray.origin(), &origin);
        assert_eq!(ray.direction(), &direction);
    }

    #[test]
    fn computing_a_point_from_distance() {
        let r = Ray::new(Point::new(2., 3., 4.), Vector::new(1., 0., 0.));
        assert_eq!(r.position(0.), Point::new(2., 3., 4.));
        assert_eq!(r.position(1.), Point::new(3., 3., 4.));
        assert_eq!(r.position(-1.), Point::new(1., 3., 4.));
        assert_eq!(r.position(2.5), Point::new(4.5, 3., 4.));
    }

    #[test]
    fn translating_a_ray() {
        let r = Ray::new(Point::new(1., 2., 3.), Vector::new(0., 1., 0.));
        let m = Transform::identity().translate(3., 4., 5.);
        let r2 = r.transform(m);
        assert_eq!(r2.origin(), &Point::new(4., 6., 8.));
        assert_eq!(r2.direction(), &Vector::new(0., 1., 0.));
    }

    #[test]
    fn scaling_a_ray() {
        let r = Ray::new(Point::new(1., 2., 3.), Vector::new(0., 1., 0.));
        let m = Transform::identity().scale(2., 3., 4.);
        let r2 = r.transform(m);
        assert_eq!(r2.origin(), &Point::new(2., 6., 12.));
        assert_eq!(r2.direction(), &Vector::new(0., 3., 0.));
    }
}
