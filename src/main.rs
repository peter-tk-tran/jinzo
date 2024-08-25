pub mod gpt2;

use candle_core::{Device, Result, Tensor};
use candle_nn::loss;
use gpt2::model::GPT2Model;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let model = GPT2Model::new(50275, 768, 0.2, &device)?;

    let token_ids: Vec<u32> = vec![0, 12, 30];
    let token_ids = Tensor::from_vec(token_ids, (1, 3), &device)?;

    // If I wanted to add an optimizer, how would that look?
    // let optimizer = AdamW::new()?;

    let pred = model.forward(&token_ids)?;
    let target_ids: Vec<u32> = vec![0, 1, 2];
    let target = Tensor::from_vec(target_ids, (3,), &device)?;
    let pred = pred.reshape((3, 50275))?;
    let loss = loss::cross_entropy(&pred, &target)?;
    println!("Loss {:?}", loss);
    let grad = loss.backward()?;
    println!("{:?}", grad);
    Ok(())
}
