use crate::point::Point;
use crate::rays::Ray;

pub struct Sphere {
    id: u32,
}

impl Sphere {
    pub fn new(id: u32) -> Self {
        Self { id }
    }
    pub fn intersect(&self, r: &Ray) -> Vec<f32> {
        let sphere_to_ray = r.origin().clone() - Point::new(0., 0., 0.);
        let a = r.direction().dot(&r.direction());
        let b = 2. * r.direction().dot(&sphere_to_ray);
        let c = sphere_to_ray.dot(&sphere_to_ray) - 1.;
        let discriminant = b * b - 4. * a * c;
        if discriminant < 0. {
            return Vec::new();
        }
        let t1 = (-b - discriminant.sqrt()) / (2. * a);
        let t2 = (-b + discriminant.sqrt()) / (2. * a);
        vec![t1, t2]
    }
}

#[cfg(test)]
mod sphere_tests {
    use crate::float_eq::f32_are_eq;
    use crate::point::Point;
    use crate::rays::Ray;
    use crate::sphere::Sphere;
    use crate::vector::Vector;

    #[test]
    fn ray_misses_sphere() {
        let r = Ray::new(Point::new(0., 2., -5.), Vector::new(0., 0., 1.));
        let s = Sphere::new(1);
        let xs = s.intersect(&r);
        assert_eq!(xs.len(), 0);
    }

    #[test]
    fn ray_intersects_at_two_points() {
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let s = Sphere::new(1);
        let xs = s.intersect(&r);
        assert_eq!(xs.len(), 2);
        assert!(f32_are_eq(xs[0], 4.));
        assert!(f32_are_eq(xs[1], 6.));
    }

    #[test]
    fn ray_intersects_at_a_tangent() {
        let r = Ray::new(Point::new(0., 1., -5.), Vector::new(0., 0., 1.));
        let s = Sphere::new(1);
        let xs = s.intersect(&r);
        assert_eq!(xs.len(), 2);
        assert!(f32_are_eq(xs[0], 5.));
        assert!(f32_are_eq(xs[1], 5.));
    }

    #[test]
    fn ray_originates_inside_a_sphere() {
        let r = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let s = Sphere::new(1);
        let xs = s.intersect(&r);
        assert_eq!(xs.len(), 2);
        assert!(f32_are_eq(xs[0], -1.));
        assert!(f32_are_eq(xs[1], 1.));
    }

    #[test]
    fn sphere_is_behind_ray() {
        let r = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
        let s = Sphere::new(1);
        let xs = s.intersect(&r);
        assert_eq!(xs.len(), 2);
        assert!(f32_are_eq(xs[0], -6.));
        assert!(f32_are_eq(xs[1], -4.));
    }
}
