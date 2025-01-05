use crate::coord::Coord;
use crate::point::Point;
use crate::vector::Vector;

mod coord;
mod float_eq;
mod point;
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

fn tick(proj: Projectile, env: Environment) -> Projectile {
    let transform = proj.transform + proj.velocity;
    let velocity = proj.velocity + env.gravity + env.winds;
    return Projectile {
        transform,
        velocity,
    };
}

fn main() {
    let mut proj = Projectile {
        transform: Point::new(0., 1., 0.),
        velocity: Vector::new(1., 1., 0.).normalize() * 2.,
    };
    let env = Environment {
        winds: Vector::new(-0.01, 0., 0.),
        gravity: Vector::new(0., -0.1, 0.),
    };

    let mut tick_count = 0;
    while proj.transform.y() > 0. {
        proj = tick(proj, env.clone());
        println!(
            "x:{}, y: {}, z: {}",
            proj.transform.x(),
            proj.transform.y(),
            proj.transform.z()
        );
        tick_count += 1;
    }
    println!("Projectile touched ground after {} ticks.", tick_count);
}
