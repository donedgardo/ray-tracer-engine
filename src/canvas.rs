use crate::color::Color;
use std::collections::HashMap;

pub struct Canvas {
    width: u32,
    height: u32,
    pixels: HashMap<(u32, u32), Color>,
}

impl Canvas {
    pub fn new(width: u32, height: u32) -> Self {
        let pixel_count: usize = (width * height) as usize;
        let mut pixels = HashMap::with_capacity(pixel_count);
        for y in 0..height {
            for x in 0..width {
                pixels.insert((x, y), Color::new(0., 0., 0.));
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn get_pixels(&self) -> impl Iterator<Item = &Color> {
        let pixel_count: usize = (self.width * self.height) as usize;
        let mut pixels = Vec::with_capacity(pixel_count);
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel_color = self.pixel_at(x, y).unwrap();
                pixels.push(pixel_color);
            }
        }
        pixels.into_iter()
    }

    pub fn write_pixel(&mut self, x: u32, y: u32, color: Color) {
        self.pixels.insert((x, y), color);
    }

    pub fn pixel_at(&self, x: u32, y: u32) -> Option<&Color> {
        self.pixels.get(&(x, y))
    }
}

#[cfg(test)]
mod canvas_tests {
    use crate::canvas::Canvas;
    use crate::color::Color;

    #[test]
    fn has_width() {
        let canvas = create_canvas();
        assert_eq!(canvas.width(), 20);
    }

    #[test]
    fn has_height() {
        let canvas = create_canvas();
        assert_eq!(canvas.height(), 40);
    }

    #[test]
    fn has_default_all_pixels_black() {
        let canvas = create_canvas();
        let mut pixels = canvas.get_pixels();
        assert!(pixels.all(|color| color == &Color::new(0., 0., 0.)))
    }

    #[test]
    fn can_write_pixels() {
        let red = Color::new(1., 0., 0.);
        let mut canvas = create_canvas();
        canvas.write_pixel(2, 3, red.clone());
        assert_eq!(canvas.pixel_at(2, 3), Some(&red));
    }

    fn create_canvas() -> Canvas {
        Canvas::new(20, 40)
    }
}
