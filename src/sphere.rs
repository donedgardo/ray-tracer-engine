use crate::intersection::Intersection;
use crate::point::Point;
use crate::rays::Ray;
use crate::transforms::matrix::Matrix4x4;
use crate::transforms::Transform;
use std::any::Any;

#[derive(PartialEq, Debug)]
pub struct Sphere {
    id: u32,
    transform: Matrix4x4,
}

impl Sphere {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            transform: Transform::identity(),
        }
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn transform(&self) -> &Matrix4x4 {
        &self.transform
    }

    pub fn set_transform(&mut self, transform: Matrix4x4) {
        self.transform = transform;
    }

    pub fn intersect(&self, r: &Ray) -> Vec<Intersection> {
        let empty = Vec::new();
        match self.transform.inverse() {
            None => empty,
            Some(inverse) => {
                let ray = r.transform(inverse);
                let sphere_to_ray = ray.origin().clone() - Point::new(0., 0., 0.);
                let a = ray.direction().dot(&ray.direction());
                let b = 2. * ray.direction().dot(&sphere_to_ray);
                let c = sphere_to_ray.dot(&sphere_to_ray) - 1.;
                let discriminant = b * b - 4. * a * c;
                if discriminant < 0. {
                    return empty;
                }
                let t1 = (-b - discriminant.sqrt()) / (2. * a);
                let t2 = (-b + discriminant.sqrt()) / (2. * a);
                vec![Intersection::new(t1, &self), Intersection::new(t2, &self)]
            }
        }
    }
}

#[cfg(test)]
mod sphere_tests {
    use crate::float_eq::f32_are_eq;
    use crate::point::Point;
    use crate::rays::Ray;
    use crate::sphere::Sphere;
    use crate::transforms::{Transform, Transformable};
    use crate::vector::Vector;

    #[test]
    fn sphere_has_id() {
        let s = Sphere::new(1);
        assert_eq!(s.id(), 1);
    }

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
        assert!(f32_are_eq(xs[0].t(), 4.));
        assert!(f32_are_eq(xs[1].t(), 6.));
    }

    #[test]
    fn ray_intersects_at_a_tangent() {
        let r = Ray::new(Point::new(0., 1., -5.), Vector::new(0., 0., 1.));
        let s = Sphere::new(1);
        let xs = s.intersect(&r);
        assert_eq!(xs.len(), 2);
        assert!(f32_are_eq(xs[0].t(), 5.));
        assert!(f32_are_eq(xs[1].t(), 5.));
    }

    #[test]
    fn ray_originates_inside_a_sphere() {
        let r = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let s = Sphere::new(1);
        let xs = s.intersect(&r);
        assert_eq!(xs.len(), 2);
        assert!(f32_are_eq(xs[0].t(), -1.));
        assert!(f32_are_eq(xs[1].t(), 1.));
    }

    #[test]
    fn sphere_is_behind_ray() {
        let r = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
        let s = Sphere::new(1);
        let xs = s.intersect(&r);
        assert_eq!(xs.len(), 2);
        assert!(f32_are_eq(xs[0].t(), -6.));
        assert!(f32_are_eq(xs[1].t(), -4.));
    }

    #[test]
    fn intersect_set_object_id() {
        let r = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
        let s = Sphere::new(1);
        let xs = s.intersect(&r);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0].object_id(), 1);
        assert_eq!(xs[1].object_id(), 1);
    }

    #[test]
    fn default_transformation() {
        let s = Sphere::new(1);
        assert_eq!(s.transform(), &Transform::identity());
    }

    #[test]
    fn changing_transformation() {
        let mut s = Sphere::new(1);
        let t = Transform::identity().translate(2., 3., 4.);
        s.set_transform(t.clone());
        assert_eq!(s.transform(), &t);
    }

    #[test]
    fn intersecting_a_translated_sphere_with_a_ray() {
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let mut s = Sphere::new(1);
        s.set_transform(Transform::new().translate(5., 0., 0.));
        let xs = s.intersect(&r);
        assert_eq!(xs.len(), 0);
    }

    #[test]
    fn intersecting_a_scaled_sphere_with_a_ray() {
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let mut s = Sphere::new(1);
        s.set_transform(Transform::new().scale(2., 2., 2.));
        let xs = s.intersect(&r);
        assert_eq!(xs.len(), 2);
        assert!(f32_are_eq(xs[0].t(), 3.0));
        assert!(f32_are_eq(xs[1].t(), 7.0));
    }
}
