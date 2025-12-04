use crate::canvas::Canvas;
use crate::color::Color;
use crate::coord::Coord;
use crate::point::Point;
use crate::ppm_image::PPM;
use crate::transforms::{Transform, Transformable};
use crate::vector::Vector;
use std::fs::File;
use std::io::Write;
use crate::intersection::{hit, Intersection};
use crate::rays::Ray;
use crate::sphere::Sphere;

mod canvas;
mod color;
mod coord;
mod float_eq;
mod intersection;
mod point;
mod ppm_image;
mod rays;
mod sphere;
mod transforms;
mod tuple;
mod vector;

struct Projectile {
    transform: Point,
    velocity: Vector,
}
#[derive(Clone)]
struct Environment {
    winds: Vector,
    gravity: Vector,
}

fn main() {
    create_projectile_image();
    draw_clock();
    draw_circle();
}

fn draw_circle() {
    let size: f32 = 240.;
    let wall_z = 10.;
    let wall_size: f32 = 7.;
    let wall_pixel_size = wall_size / size;
    let ray_origin = Point::new(0., 0., -10.);
    let half = wall_size / 2.;
    let mut canvas = Canvas::new(size as u32, size as u32);
    let mut sphere = Sphere::new(1);
    for y in 0..(size as usize) {
        for x in 0..(size as usize) {
            let world_y = half - wall_pixel_size * y as f32;
            let world_x = -half + wall_pixel_size * x as f32;
            let position = Point::new(world_x, world_y, wall_z);
            let vector: Vector = (position - ray_origin.clone()).into();
            let ray = Ray::new(ray_origin.clone(), vector);
            let xs = sphere.intersect(&ray);
            // refactor Idea: I think hit should be a method of xs (new struct for vec<intersection>)
            let h = hit(&xs);
            match h {
                None => {}
                Some(_) => {
                    canvas.write_pixel(x as u32, y as u32, Color::new(1., 0., 0.));
                }
            }

        }
    }
    let image_output = PPM::generate(&canvas);
    let mut file = File::create("circle.ppm").unwrap();
    file.write_all(image_output.as_bytes()).unwrap();
}

fn draw_clock() {
    let size: f32 = 120.;
    let mut canvas = Canvas::new(size as u32, size as u32);
    let hand_length: f32 = size * (3. / 8.);
    for hour in 0..12 {
        let hour_angle = std::f32::consts::FRAC_PI_6 * hour as f32;
        let hour_hand_end = Point::new(0., 0., 1.);
        let transform = Transform::identity().rotate_y(hour_angle);
        let point = transform * hour_hand_end;
        let point = Point::new(point.x() * hand_length, point.y(), point.z() * hand_length);
        let point = point + Vector::new(size / 2., 0., size / 2.);
        canvas.write_pixel(point.x() as u32, point.z() as u32, Color::new(1., 0., 0.));
    }
    let image_output = PPM::generate(&canvas);
    let mut file = File::create("clock.ppm").unwrap();
    file.write_all(image_output.as_bytes()).unwrap();
}

fn create_projectile_image() {
    let mut proj = Projectile {
        transform: Point::new(0., 1., 0.),
        velocity: Vector::new(1., 1.8, 0.).normalize() * 11.25,
    };
    let env = Environment {
        winds: Vector::new(-0.01, 0., 0.),
        gravity: Vector::new(0., -0.1, 0.),
    };
    let mut canvas = Canvas::new(900, 550);
    let mut tick_count = 0;

    while proj.transform.y() > 0. {
        proj = tick(proj, &env);
        tick_count += 1;
        canvas.write_pixel(
            proj.transform.x() as u32,
            550 - proj.transform.y() as u32,
            Color::new(1., 0., 0.),
        );
    }
    println!("Projectile touched ground after {} ticks.", tick_count);

    let image_output = PPM::generate(&canvas);
    let mut file = File::create("projectile.ppm").unwrap();
    file.write_all(image_output.as_bytes()).unwrap();
}

fn tick(proj: Projectile, env: &Environment) -> Projectile {
    let transform = proj.transform + proj.velocity;
    let velocity = proj.velocity + env.gravity + env.winds;
    return Projectile {
        transform,
        velocity,
    };
}
