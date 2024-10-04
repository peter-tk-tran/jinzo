pub mod gpt2;
pub mod lr_scheduler;
use rand::Rng;

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};
use gpt2::model::GPT2Model;
use tokenizers::tokenizer::Tokenizer;

use self::lr_scheduler::WarmupLRScheduler;

fn load_weights(vm: &mut VarMap, model_path: &Path) {
    if let Err(e) = vm.load(model_path) {
        eprintln!("Failed to load the model: {:?}", e);
    } else {
        println!("Model loaded successfully!");
    }
}

fn get_batch(
    passage: &str,
    batch_size: usize,
    context_length: usize,
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let passage_encoding = tokenizer.encode(passage, true).unwrap();
    let passage_ids = passage_encoding.get_ids();
    let mut inputs: Vec<Tensor> = Vec::with_capacity(batch_size);
    let mut labels: Vec<Tensor> = Vec::with_capacity(batch_size);
    let range = 0..passage_ids.len().saturating_sub(context_length + 1);
    for _ in 0..batch_size {
        let i = rand::thread_rng().gen_range(range.clone());
        // let i = 0;
        let input = &passage_ids[i..i + context_length];
        let label = &passage_ids[i + 1..i + 1 + context_length];

        let input = Tensor::from_vec(input.to_vec(), (1, context_length), device)?;
        let label = Tensor::from_vec(label.to_vec(), (1, context_length), device)?;

        inputs.push(input);
        labels.push(label);
    }

    let inputs = Tensor::stack(&inputs, 0)?.squeeze(1)?;
    let labels = Tensor::stack(&labels, 0)?.squeeze(1)?;

    Ok((inputs, labels))
}

fn main() -> Result<()> {
    // let model_path = Path::new("my_model");
    let model_path = Path::new("/Users/I747624/Documents/jinzo/model.safetensors");
    // let device = Device::Cpu;
    let device = Device::new_metal(0)?;
    let mut vm = VarMap::new();

    let vb = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
    let context_length = 1024;
    let vocab_size = 50257;
    let mut model = GPT2Model::new(vb)?;
    model.eval();

    if model_path.exists() {
        load_weights(&mut vm, &model_path);
    }

    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
    if !model.train_mode {
        let initial_text = "My name is Teven and I am";
        print!("{}", initial_text);
        io::stdout().flush().expect("Failed to flush stdout");

        let encoding = tokenizer.encode(initial_text, true).unwrap();
        let mut token_ids = encoding.get_ids().to_vec(); // Vec<u32>

        for _ in 0..100 {
            let input_ids = Tensor::from_vec(token_ids.clone(), (1, token_ids.len()), &device)?;
            let pred = model.forward(&input_ids)?;

            let next_token_logits = pred.i((.., token_ids.len() - 1, ..))?.squeeze(0)?; // Shape: [vocab_size]
            let next_token_id: u32 = next_token_logits.argmax(0)?.to_scalar().unwrap();
            token_ids.push(next_token_id);

            let next_token = tokenizer.decode(&[next_token_id], false).unwrap();
            print!("{}", next_token);
            io::stdout().flush().expect("Failed to flush stdout");
        }
    }
    if model.train_mode {
        let batch_size = 8;
        let learning_rate = 0.001;

        // let mut lr_scheduler = LRScheduler::new(learning_rate, decay_rate); // initial_lr = 0.001, decay_rate = 0.99
        let mut lr_scheduler = WarmupLRScheduler::new(0.001, 0.999, 10);

        let adamw_params = candle_nn::ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        };
        let mut optimizer = candle_nn::AdamW::new(vm.all_vars(), adamw_params)?;
        let passage_path = Path::new("/Users/I747624/Documents/jinzo/on_writing_well.txt");
        let mut passage_file = File::open(passage_path).unwrap();
        let mut passage = String::new();

        passage_file.read_to_string(&mut passage).unwrap();
        for _step in 0..100 {
            let (inputs, labels) =
                get_batch(&passage, batch_size, context_length, &tokenizer, &device)?;

            let logits = model.forward(&inputs)?;
            let logits = logits.reshape((batch_size * context_length, vocab_size))?;
            let labels = labels.reshape((batch_size * context_length,))?;

            let max_logits = logits.max_keepdim(1)?;
            let logits = logits.broadcast_sub(&max_logits)?;

            let loss = loss::cross_entropy(&logits, &labels)?;

            let grads = loss.backward()?;

            lr_scheduler.apply(&mut optimizer);
            optimizer.step(&grads).unwrap();

            println!(
                "Loss: {:.5?}, LR: {:.6?}",
                loss.to_scalar::<f32>()?,
                optimizer.learning_rate()
            );
            vm.save(model_path).unwrap();
        }
    }
    Ok(())
}
