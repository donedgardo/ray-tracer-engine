use crate::sphere::Sphere;

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
