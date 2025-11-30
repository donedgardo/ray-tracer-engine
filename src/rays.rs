use crate::point::Point;
use crate::vector::Vector;

struct Ray {
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
}

#[cfg(test)]
mod rays_tests {
    use crate::point::Point;
    use crate::rays::Ray;
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
}
