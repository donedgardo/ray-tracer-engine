use crate::sphere::Sphere;

#[derive(Debug, PartialEq)]
pub struct Intersection {
    t: f32,
    object_id: u32,
}

impl Intersection {
    pub fn new(t: f32, object: &Sphere) -> Self {
        Self {
            t,
            object_id: object.id(),
        }
    }
    pub fn t(&self) -> f32 {
        self.t
    }

    pub fn object_id(&self) -> u32 {
        self.object_id
    }
}

pub fn hit(intersections: Vec<&Intersection>) -> Option<&Intersection> {
    let hit = None;
    if intersections.is_empty() {
        return hit;
    }
    let mut positive_intersections = intersections
        .iter()
        .filter(|i| i.t() >= 0.)
        .collect::<Vec<&&Intersection>>();
    if positive_intersections.is_empty() {
        return hit;
    }
    positive_intersections
        .sort_by(|a, b| a.t().partial_cmp(&b.t()).expect("Tried to compare a NaN"));
    Some(positive_intersections[0])
}

#[cfg(test)]
mod hits_tests {
    use crate::intersection::{hit, Intersection};
    use crate::sphere::Sphere;

    #[test]
    fn when_all_intersections_are_positive() {
        let s = Sphere::new(1);
        let i1 = Intersection::new(1., &s);
        let i2 = Intersection::new(2., &s);
        let xs = vec![&i1, &i2];
        let i = hit(xs).unwrap();
        assert_eq!(&i1, i);
    }
    #[test]
    fn when_some_intersections_are_negative() {
        let s = Sphere::new(1);
        let i1 = Intersection::new(-1., &s);
        let i2 = Intersection::new(1., &s);
        let xs = vec![&i1, &i2];
        let i = hit(xs).unwrap();
        assert_eq!(&i2, i);
    }

    #[test]
    fn when_all_intersections_are_negative() {
        let s = Sphere::new(1);
        let i1 = Intersection::new(-2., &s);
        let i2 = Intersection::new(-1., &s);
        let xs = vec![&i1, &i2];
        let i = hit(xs);
        assert_eq!(None, i);
    }
    #[test]
    fn is_always_the_lowest_negative_intersection() {
        let s = Sphere::new(1);
        let i1 = Intersection::new(5., &s);
        let i2 = Intersection::new(7., &s);
        let i3 = Intersection::new(-3., &s);
        let i4 = Intersection::new(2., &s);
        let xs = vec![&i1, &i2, &i3, &i4];
        let i = hit(xs).unwrap();
        assert_eq!(&i4, i);
    }
}

#[cfg(test)]
mod intersection_tests {
    use crate::intersection::Intersection;
    use crate::sphere::Sphere;

    #[test]
    fn encapsulates_t_and_object() {
        let s = Sphere::new(1);
        let i = Intersection::new(3.5, &s);
        assert_eq!(i.t(), 3.5);
        assert_eq!(i.object_id(), s.id());
    }
}
