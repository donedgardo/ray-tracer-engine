use crate::canvas::Canvas;
use crate::color::Color;
use crate::coord::Coord;
use crate::point::Point;
use crate::ppm_image::PPM;
use crate::transforms::{Transform, Transformable};
use crate::vector::Vector;
use std::fs::File;
use std::io::Write;

mod canvas;
mod color;
mod coord;
mod float_eq;
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
    draw_clock();
    create_projectile_image();
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
