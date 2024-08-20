pub mod gpt2;

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module};

struct Model {
    l1: Linear,
    l2: Linear,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = self.l1.forward(image)?;
        let x = x.relu()?;
        let x = self.l2.forward(&x)?;
        Ok(x)
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let w1 = Tensor::randn(0f32, 1.0, (100, 784), &device)?;
    let b1 = Tensor::randn(0f32, 1.0, (100,), &device)?;
    // let l1 = Linear::new(w1, Some(b1));

    let w2 = Tensor::randn(0f32, 1.0, (10, 100), &device)?;
    let b2 = Tensor::randn(0f32, 1.0, (10,), &device)?;
    // let l2 = Linear::new(w2, Some(b2));

    // let model = Model { l1, l2 };

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;
    // let digit = model.forward(&dummy_image)?;
    // println!("Digit {digit:?} digit");

    let x = dummy_image.matmul(&w1.t()?)?;
    let x = b1.broadcast_add(&x)?;

    let x = x.matmul(&w2.t()?)?;
    let x = x.broadcast_add(&b2)?;
    println!("{x:?}");

    Ok(())
}
