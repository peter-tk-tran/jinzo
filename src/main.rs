pub mod gpt2;

use std::path::Path;

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{loss, VarBuilder, VarMap};
use gpt2::model::GPT2Model;
use tokenizers::tokenizer::Tokenizer;

fn count_params(vm: &VarMap) {
    let mut num_params = 0;
    for var in vm.all_vars() {
        num_params += var.elem_count();
    }
    println!("Total Params: {num_params}");
}

fn load_weights(vm: &mut VarMap, model_path: &Path) {
    if let Err(e) = vm.load(model_path) {
        eprintln!("Failed to load the model: {:?}", e);
    } else {
        println!("Model loaded successfully!");
    }
}

fn main() -> Result<()> {
    let model_path = Path::new("my_model");
    let device = Device::Cpu;
    let mut vm = VarMap::new();

    let vb = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
    let vocab_size = 50257;
    let model = GPT2Model::new(vocab_size, 768, 0.2, vb)?;
    load_weights(&mut vm, &model_path);

    // count_params(&vm);

    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
    let encoding: tokenizers::Encoding = tokenizer.encode("Hello how are you?", false).unwrap();
    let token_ids = Tensor::from_vec(encoding.get_ids().to_vec(), (1, encoding.len()), &device)?;

    println!("{}", token_ids);
    // If I wanted to add an optimizer, how would that look?
    // let optimizer = AdamW::new()?;

    let pred = model.forward(&token_ids)?;
    let next_token_logits = pred.i((.., encoding.len() - 1, ..))?;
    println!("Next Token Logits: {next_token_logits}");

    let next_token_id = next_token_logits.argmax(1)?;
    let next_token = tokenizer.decode(&next_token_id.to_vec1()?, false).unwrap();
    println!("Next Token: {next_token}");

    // let model_path = Path::new("my_model");
    // vm.save(model_path).unwrap();

    // println!("{pred}");
    // let target_ids: Vec<u32> = vec![0, 1, 2];
    // let target = Tensor::from_vec(target_ids, (3,), &device)?;
    // let pred = pred.reshape((3, vocab_size))?;
    // let loss = loss::cross_entropy(&pred, &target)?;
    // println!("Loss {:?}", loss);
    // let grad = loss.backward()?;
    // println!("{:?}", grad);

    // println!("{:?}", vm.data());
    // println!("{:?}", vm.all_vars());
    Ok(())
}
