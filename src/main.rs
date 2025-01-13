use crate::canvas::Canvas;
use crate::color::Color;
use crate::coord::Coord;
use crate::matrix::Matrix4x4;
use crate::point::Point;
use crate::ppm_image::PPM;
use crate::tuple::Tuple;
use crate::vector::Vector;
use std::fs::File;
use std::io::Write;
use std::ops::{Index, Mul};

mod canvas;
mod color;
mod coord;
mod float_eq;
mod matrix;
mod point;
mod ppm_image;
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

fn tick(proj: Projectile, env: &Environment) -> Projectile {
    let transform = proj.transform + proj.velocity;
    let velocity = proj.velocity + env.gravity + env.winds;
    return Projectile {
        transform,
        velocity,
    };
}

fn main() {
    // create_projectile_image();
    let m = Matrix4x4::new(
        [1., 0., 0., 0.],
        [2., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    );

    let t = Tuple::new(1., 2., 3., 4.);

    let c = m * t;

    println!("{:?}", &c);
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
    let mut file = File::create("projectile.pmp").unwrap();
    file.write_all(image_output.as_bytes()).unwrap();
}
