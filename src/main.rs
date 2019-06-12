extern crate image;
#[macro_use]
extern crate lazy_static;
#[macro_use(array, s)]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate rand;
extern crate rayon;

use ndarray::{Array1, Array3};
use ndarray_linalg::norm::Norm;

type Color = Array1<u8>;
type Pixels = Array3<u8>;
type Voxels = Array3<i32>;

struct Camera {
    position: [f32; 3],
    view_dir: [f32; 3],
    fov: f32,
}

impl Camera {
    fn new(position: [f32; 3], view_dir: [f32; 3]) -> Camera {
        Camera {
            position,
            view_dir,
            fov: std::f32::consts::FRAC_PI_2,
        }
    }
}

struct Material {
    color: Color,
}

impl Material {
    fn new(color: Color) -> Material {
        Material { color }
    }
}

lazy_static! {
    static ref MATERIALS: [Material; 3] = [
        Material::new(array!(255, 0, 0)),
        Material::new(array!(0, 255, 0)),
        Material::new(array!(0, 0, 255)),
    ];
}

fn set_voxels(voxels: &mut Voxels, start: [usize; 3], end: [usize; 3], value: i32) {
    for i in start[0]..end[0] {
        for j in start[1]..end[1] {
            for k in start[2]..end[2] {
                voxels[[i, j, k]] = value;
            }
        }
    }
}

fn march<F>(from: &[f32; 3], direction: &[f32; 3], mut voxel_fn: F)
where
    F: FnMut([i32; 3], f32) -> bool,
{
    // Parse out the starting position components.
    let [x, y, z] = [from[0], from[1], from[2]];

    // Get the sign bit of each direction component.
    let [sx, sy, sz] = [
        direction[0].is_sign_negative(),
        direction[1].is_sign_negative(),
        direction[2].is_sign_negative(),
    ];

    // Get the distance partials of each direction component.
    let norm = array!(direction[0], direction[1], direction[2]).norm();
    let [dx, dy, dz] = [
        norm / direction[0].abs(),
        norm / direction[1].abs(),
        norm / direction[2].abs(),
    ];

    // The ray distance to the next intersection in each direction.
    let [mut dist_x, mut dist_y, mut dist_z] = [
        dx * if sx {
            x - x.floor()
        } else {
            1.0 + x.floor() - x
        },
        dy * if sy {
            y - y.floor()
        } else {
            1.0 + y.floor() - y
        },
        dz * if sz {
            z - z.floor()
        } else {
            1.0 + z.floor() - z
        },
    ];

    // March over the integer voxel cells until the voxel fn returns false.
    let mut distance = 0.0;
    let [mut ix, mut iy, mut iz] = [x as i32, y as i32, z as i32];
    loop {
        if !voxel_fn([ix, iy, iz], distance) {
            break;
        }

        // Advance one voxel in the direction of nearest intersection.
        if dist_x <= dist_y && dist_x <= dist_z {
            distance += dist_x;
            ix += if sx { -1 } else { 1 };
            dist_x += dx;
        } else if dist_y <= dist_z {
            distance += dist_y;
            iy += if sy { -1 } else { 1 };
            dist_y += dy;
        } else {
            distance += dist_z;
            iz += if sz { -1 } else { 1 };
            dist_z += dz;
        }
    }
}

fn main() {
    let voxels_size = 128;
    let mut voxels = Voxels::zeros((voxels_size, voxels_size, voxels_size));

    // Initialize voxels.
    set_voxels(&mut voxels, [0, 0, 0], [1, 128, 128], 1);
    set_voxels(&mut voxels, [0, 0, 0], [128, 1, 128], 2);
    set_voxels(&mut voxels, [0, 0, 0], [128, 128, 1], 3);
    set_voxels(&mut voxels, [127, 0, 0], [128, 128, 128], 1);
    set_voxels(&mut voxels, [0, 127, 0], [128, 128, 128], 2);
    set_voxels(&mut voxels, [0, 0, 127], [128, 128, 128], 3);

    // Initialize pixels.
    let pixels_size = 1024;

    // Initialize the camera.
    let camera = Camera::new([60.0, 64.0, 64.0], [1.0, 0.0, 0.0]);

    // Prepare the ray directions for each pixel.
    let timer = std::time::Instant::now();
    let mut rays = Array3::<f32>::zeros((pixels_size, pixels_size, 3));
    for i in 0..pixels_size {
        for j in 0..pixels_size {
            let x = (i as f32 + 0.5) / pixels_size as f32;
            let y = (j as f32 + 0.5) / pixels_size as f32;
            let theta = camera.fov * (x - 0.5);
            let phi = camera.fov * (y - 0.5);
            rays[[i, j, 0]] = theta.cos() * phi.cos();
            rays[[i, j, 1]] = theta.sin() * phi.cos();
            rays[[i, j, 2]] = phi.sin();
        }
    }
    println!("{}", timer.elapsed().as_millis());

    // March a ray through each pixel to compute its color.
    use rayon::prelude::*;
    let indices: Vec<usize> = (0..pixels_size * pixels_size).collect();
    let timer = std::time::Instant::now();
    let pixels: Vec<[u8; 3]> = indices
        .par_iter()
        .map(|index| -> [u8; 3] {
            let i = index % pixels_size;
            let j = index / pixels_size;
            let ray = [rays[[i, j, 0]], rays[[i, j, 1]], rays[[i, j, 2]]];

            let mut pixel = [0, 0, 0];
            march(&camera.position, &ray, |[ix, iy, iz], _dist| -> bool {
                if ix < 0 || iy < 0 || iz < 0 {
                    return false;
                }

                let [vx, vy, vz] = [ix as usize, iy as usize, iz as usize];
                if vx >= voxels_size || vy >= voxels_size || vz >= voxels_size {
                    return false;
                }

                let mat = voxels[[vx, vy, vz]];
                if mat > 0 {
                    let color = &MATERIALS[(mat - 1) as usize].color;
                    pixel[0] = color[0];
                    pixel[1] = color[1];
                    pixel[2] = color[2];
                    return false;
                }

                true
            });

            pixel
        })
        .collect();

    println!("{}", timer.elapsed().as_millis());

    // Create an image for now by simple setting random pixel colors.
    let mut img = image::ImageBuffer::new(pixels_size as u32, pixels_size as u32);
    for (index, color) in pixels.iter().enumerate() {
        let i = index % pixels_size;
        let j = index / pixels_size;
        img.put_pixel(
            i as u32,
            j as u32,
            image::Rgb([color[0], color[1], color[2]]),
        );
    }
    img.save("test.png").unwrap();
}
