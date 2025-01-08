use crate::canvas::Canvas;
use crate::color::Color;

pub struct PPM;

impl PPM {
    pub fn generate(canvas: &Canvas) -> String {
        let header = Self::generate_header(canvas);
        let body = Self::generate_body(canvas);
        header + &body
    }

    fn generate_body(canvas: &Canvas) -> String {
        let mut body = "".to_string();
        for y in 0..canvas.height() {
            let mut row = "\n".to_string();
            for x in 0..canvas.width() {
                let color = canvas.pixel_at(x, y).unwrap();
                row.push_str(Self::generate_x_row(canvas, x, &color).as_str());
            }
            row = Self::truncate_rows(&row, 70);
            body.push_str(&row);
        }
        body.push_str("\n");
        body
    }

    fn truncate_rows(row: &String, max_line_size: usize) -> String {
        let mut truncated_rows = row.clone();
        let row_char_count = row.chars().count();
        if row_char_count > max_line_size {
            for i in 1..=(row_char_count / max_line_size) {
                let split_until = max_line_size - 1;
                let split = &truncated_rows[..split_until * i];
                let last_space_index = split.rfind(" ");
                if let Some(mid) = last_space_index {
                    let (a, b) = truncated_rows.split_at(mid + 1);
                    truncated_rows =
                        format!("{}\n{}", a.trim_end_matches(' '), b.trim_end_matches(' '));
                }
            }
        }
        truncated_rows
    }

    fn generate_x_row(canvas: &Canvas, x: u32, color: &&Color) -> String {
        let mut row = "".to_string();
        row.push_str(&format!(
            "{} {} {}",
            Self::get_255_value(color.red()),
            Self::get_255_value(color.green()),
            Self::get_255_value(color.blue()),
        ));
        if x != canvas.width() - 1 {
            row.push_str(" ");
        }
        row
    }

    fn get_255_value(color: f32) -> u32 {
        (color * 255.0).clamp(0., 255.).ceil() as u32
    }

    fn generate_header(canvas: &Canvas) -> String {
        format!(
            "P3\n\
            {} {}\n\
            255",
            canvas.width(),
            canvas.height()
        )
    }
}

#[cfg(test)]
mod ppm_tests {
    use crate::canvas::Canvas;
    use crate::color::Color;
    use crate::ppm_image::PPM;

    #[test]
    fn constructs_header() {
        let canvas = Canvas::new(10, 20);
        let output = PPM::generate(&canvas);
        let lines: Vec<_> = output.split("\n").collect();
        let header_output = lines[0..=2].join("\n");

        let expected_output = "P3\n\
                                   10 20\n\
                                   255";
        assert_eq!(header_output, expected_output)
    }

    #[test]
    fn generates_pixel_data() {
        let mut canvas = Canvas::new(5, 3);
        let c1 = Color::new(1.5, 0., 0.);
        canvas.write_pixel(0, 0, c1);
        let c2 = Color::new(0., 0.5, 0.);
        canvas.write_pixel(2, 1, c2);
        let c3 = Color::new(-0.5, 0., 1.);
        canvas.write_pixel(4, 2, c3);
        let output = PPM::generate(&canvas);
        let lines: Vec<_> = output.split("\n").collect();
        let pixel_data_output = lines[3..=5].join("\n");
        let expected_output = "255 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
                                    0 0 0 0 0 0 0 128 0 0 0 0 0 0 0\n\
                                    0 0 0 0 0 0 0 0 0 0 0 0 0 0 255";
        assert_eq!(pixel_data_output, expected_output)
    }

    #[test]
    fn splits_long_lines() {
        let mut canvas = Canvas::new(10, 2);
        for y in 0..canvas.height() {
            for x in 0..canvas.width() {
                canvas.write_pixel(x, y, Color::new(1., 0.8, 0.6));
            }
        }
        let ppm = PPM::generate(&canvas);
        println!("{}", ppm);
        let lines: Vec<_> = ppm.split("\n").collect();
        let pixel_data_output = lines[3..=6].join("\n");
        let expected_output =
            "255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204\n\
            153 255 204 153 255 204 153 255 204 153 255 204 153\n\
            255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204\n\
            153 255 204 153 255 204 153 255 204 153 255 204 153";

        assert_eq!(pixel_data_output, expected_output);
    }

    #[test]
    fn splits_long_lines2() {
        let mut canvas = Canvas::new(40, 2);
        for y in 0..canvas.height() {
            for x in 0..canvas.width() {
                canvas.write_pixel(x, y, Color::new(0.047, 0.047, 0.047));
            }
        }

        let ppm = PPM::generate(&canvas);
        let lines = ppm.lines();
        for line in lines {
            assert!(line.chars().count() <= 70)
        }
    }

    #[test]
    fn splits_long_lines1() {
        let canvas = Canvas::new(40, 2);
        let ppm = PPM::generate(&canvas);
        let lines = ppm.lines();
        for line in lines {
            assert!(line.chars().count() <= 70)
        }
    }

    #[test]
    fn ends_with_new_line() {
        let canvas = Canvas::new(40, 2);
        let ppm = PPM::generate(&canvas);
        assert_eq!(ppm.chars().last(), Some('\n'))
    }
}
