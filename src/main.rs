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
    let model_path = Path::new("/Users/petertran/Desktop/model.safetensors");
    // let device = Device::Cpu;
    let device = Device::new_metal(0)?;
    let mut vm = VarMap::new();

    let vb = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
    let context_length = 1024;
    let vocab_size = 50257;
    let embed_dim = 768;
    let dropout_p = 0.2;
    let mut model = GPT2Model::new(vocab_size, context_length, embed_dim, dropout_p, vb)?;
    model.train = false;

    let model_tensors = candle_core::safetensors::load(model_path, &device)?;
    for (k, _v) in model_tensors.iter() {
        println!("Key {k}");
        // println!("Value {v}");
    }

    if model_path.exists() {
        load_weights(&mut vm, &model_path);
    }

    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
    if !model.train {
        let mut query = "Abraham Lincoln is".to_string();
        for _ in 0..100 {
            let encoding: tokenizers::Encoding = tokenizer.encode(query.clone(), true).unwrap();
            let encoding_ids = encoding.get_ids().to_vec();
            let token_ids = Tensor::from_vec(encoding_ids, (1, encoding.len()), &device)?;
            let pred = model.forward(&token_ids)?;
            let next_token_logits = pred.i((.., encoding.len() - 1, ..))?;
            // println!("{next_token_logits}");
            let next_token_id = next_token_logits.argmax(1)?;
            let next_token = tokenizer.decode(&next_token_id.to_vec1()?, true).unwrap();
            print!("{next_token}");
            io::stdout().flush().expect("Failed to flush stdout");
            query.push_str(&next_token);
        }
    }
    if model.train {
        let batch_size = 8;
        let learning_rate = 0.001;

        // let mut lr_scheduler = LRScheduler::new(learning_rate, decay_rate); // initial_lr = 0.001, decay_rate = 0.99
        let mut lr_scheduler = WarmupLRScheduler::new(0.001, 0.999, 10);

        let adamw_params = candle_nn::ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        };
        let mut optimizer = candle_nn::AdamW::new(vm.all_vars(), adamw_params)?;
        let passage_path =
            Path::new("/Users/petertran/Documents/projects/jinzo/on_writing_well.txt");
        let mut passage_file = File::open(passage_path).unwrap();
        let mut passage = String::new();

        passage_file.read_to_string(&mut passage).unwrap();
        // println!("{passage}");
        for _step in 0..100 {
            let (inputs, labels) =
                get_batch(&passage, batch_size, context_length, &tokenizer, &device)?;

            // println!("Labels shape: {:?}", labels.shape());
            // println!("Labels: {:?}", labels);

            // println!("{input} {label}");
            // println!(
            //     "Input Shape: {:?}, Label Shape: {:?}",
            //     input.shape(),
            //     label.shape()
            // );
            let logits = model.forward(&inputs)?;
            // println!("Raw {logits}");
            let logits = logits.reshape((batch_size * context_length, vocab_size))?;
            // println!("After Reshape {logits}");
            let labels = labels.reshape((batch_size * context_length,))?;

            let max_logits = logits.max_keepdim(1)?;
            let logits = logits.broadcast_sub(&max_logits)?;

            let loss = loss::cross_entropy(&logits, &labels)?;
            // let probs = candle_nn::ops::softmax(&logits, 1)?;
            // // println!("Probs {probs}");

            // let loss = loss::nll(&probs, &labels)?;
            // // println!("Calculated Loss! {loss}");

            let grads = loss.backward()?;
            // // println!("Got grads {grads:?}");

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
    // let model_path = Path::new("my_model");

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
