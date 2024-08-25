pub mod gpt2;

use core::f32;
use core::ops::Sub;

use candle_core::{Device, IndexOp, Result, Shape, Tensor};
use candle_nn::ops::softmax;
use candle_nn::{Activation, Conv1d, Conv1dConfig, Dropout, Embedding, LayerNorm, Linear, Module};

pub struct Attention {
    num_heads: usize,
    head_dim: usize,
    c_attn: Conv1d, // Concatenated attention
    c_proj: Conv1d, // Concatenated projection, instead of concating the results of each head, mix
    device: Device, // them.
}
impl Attention {
    fn new(num_heads: usize, embed_dim: usize, device: &Device) -> Result<Self> {
        // let (c_out, c_in_k, k_size) = kernel.dims3()?;
        let c_attn_weight = Tensor::rand(0f32, 1.0, (embed_dim * 3, embed_dim, 1), device)?;
        let c_attn_bias = Tensor::rand(0f32, 1.0, (embed_dim * 3,), device)?;
        let c_attn_config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let c_attn = Conv1d::new(c_attn_weight, Some(c_attn_bias), c_attn_config);

        let c_proj_weight = Tensor::rand(0f32, 1.0, (embed_dim, embed_dim, 1), device)?;
        let c_proj_bias = Tensor::rand(0f32, 1.0, (embed_dim,), device)?;
        let c_proj_config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let c_proj = Conv1d::new(c_proj_weight, Some(c_proj_bias), c_proj_config);

        let head_dim = embed_dim / num_heads;
        Ok(Self {
            num_heads,
            head_dim,
            c_attn,
            c_proj,
            device: device.clone(),
        })
    }
}
impl Module for Attention {
    fn forward(&self, hidden_state: &Tensor) -> Result<Tensor> {
        let (batch_size, num_tokens, embed_dim) = hidden_state.dims3()?;
        // let hidden_state = hidden_state.transpose(1, 2)?;
        let kqv = self.c_attn.forward(&hidden_state.transpose(1, 2)?)?;
        let kqv = kqv.transpose(1, 2)?; // Shape: B, T, embed_dim * 3
        let kqv = kqv.reshape((batch_size, num_tokens, 3, embed_dim))?;

        let q = kqv.i((.., .., 0, ..))?; // batch_size, num_tokens, embed_dim
        let k = kqv.i((.., .., 1, ..))?;
        let v = kqv.i((.., .., 2, ..))?;

        let org_shape = q.shape();
        let new_shape = Shape::from_dims(&[batch_size, num_tokens, self.num_heads, self.head_dim]);

        let q = q.reshape(&new_shape)?;
        let k = k.reshape(&new_shape)?;
        let v = v.reshape(&new_shape)?;

        let q = q.permute((0, 2, 1, 3))?; // b, heads, seq_length, head_features
        let k = k.permute((0, 2, 1, 3))?;
        let v = v.permute((0, 2, 1, 3))?;

        let inf = Tensor::full(f32::MAX, (num_tokens, num_tokens), &self.device)?;
        let tril = Tensor::tril2(num_tokens, candle_core::DType::F32, &self.device)?.sub(1.0)?;
        let causal_mask = inf.broadcast_mul(&tril)?;

        let weights = q.matmul(&k.transpose(2, 3)?)?;
        let weights = weights.broadcast_add(&causal_mask)?; // do not let previous tokens see the future

        let weights = softmax(&weights, 3)?; // weight shape (B, heads, seq_length, seq_length)

        // println!("{v}"); // 1, 12, 4, 64
        // println!("{weights}");
        // TODO add dropout to the weights!
        let attn = weights.matmul(&v)?; // (B, heads, seq_length, head_features)
                                        //
        let attn = attn.reshape(org_shape)?;
        let attn = self.c_proj.forward(&attn.transpose(1, 2)?)?;
        let attn = attn.transpose(1, 2)?;
        // TODO add drop out after the forward as well!
        Ok(attn)
    }
}

pub struct GPT2MLP {
    c_fc: Conv1d,
    c_proj: Conv1d,
    c_act: Activation,
    c_dropout: Dropout,
}
impl GPT2MLP {
    pub fn new(intermediate_size: usize, embed_dim: usize, device: &Device) -> Result<Self> {
        let c_fc_weight = Tensor::rand(0f32, 1.0, (intermediate_size, embed_dim, 1), device)?;
        let c_fc_bias = Tensor::rand(0f32, 1.0, (intermediate_size,), device)?;
        let c_fc_config = Conv1dConfig {
            padding: 0,
            dilation: 1,
            groups: 1,
            stride: 1,
        };
        let c_fc = Conv1d::new(c_fc_weight, Some(c_fc_bias), c_fc_config);

        let c_proj_weight = Tensor::rand(0f32, 1.0, (embed_dim, intermediate_size, 1), device)?;
        let c_proj_bias = Tensor::rand(0f32, 1.0, (embed_dim,), device)?;
        let c_proj_config = Conv1dConfig {
            padding: 0,
            dilation: 1,
            groups: 1,
            stride: 1,
        };
        let c_proj = Conv1d::new(c_proj_weight, Some(c_proj_bias), c_proj_config);

        let c_act = Activation::Gelu;
        let c_dropout = Dropout::new(0.2);

        Ok(GPT2MLP {
            c_fc,
            c_proj,
            c_act,
            c_dropout,
        })
    }
}
impl Module for GPT2MLP {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.c_fc.forward(&hidden_states.transpose(1, 2)?)?;
        let hidden_states = hidden_states.transpose(1, 2)?;
        let hidden_states = self.c_act.forward(&hidden_states)?;
        let hidden_states = self.c_proj.forward(&hidden_states.transpose(1, 2)?)?;
        let hidden_states = hidden_states.transpose(1, 2)?;
        let hidden_states = self.c_dropout.forward(&hidden_states, true)?; // SET TRAIN TO FALSE
        Ok(hidden_states)
    }
}

pub struct GPT2Block {
    ln_1: LayerNorm,
    attn: Attention,
    ln_2: LayerNorm,
    mlp: GPT2MLP,
}
impl GPT2Block {
    fn new(embed_dim: usize, device: &Device) -> Result<Self> {
        let ln_1_tensor = Tensor::rand(0f32, 1.0, (embed_dim,), device)?;
        let ln_1_bias = Tensor::rand(0f32, 1.0, (embed_dim,), device)?;
        let ln_1_eps = 1e-05;
        let ln_1 = LayerNorm::new(ln_1_tensor, ln_1_bias, ln_1_eps);

        let num_heads = 12;
        let attn = Attention::new(num_heads, embed_dim, device)?;

        let ln_2_tensor = Tensor::rand(0f32, 1.0, (embed_dim,), device)?;
        let ln_2_bias = Tensor::rand(0f32, 1.0, (embed_dim,), device)?;
        let ln_2_eps = 1e-05;
        let ln_2 = LayerNorm::new(ln_2_tensor, ln_2_bias, ln_2_eps);

        let mlp = GPT2MLP::new(4 * embed_dim, embed_dim, device)?;

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }
}

impl Module for GPT2Block {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();

        let hidden_states = self.ln_1.forward(hidden_states)?;

        let attn = self.attn.forward(&hidden_states)?;

        let hidden_states = residual.add(&attn)?;

        let residual = hidden_states.clone();
        let hidden_states = self.ln_2.forward(&hidden_states)?;

        let ff_hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = residual.add(&ff_hidden_states)?;

        Ok(hidden_states)
    }
}

pub struct GPT2Model {
    wte: Embedding,
    wpe: Embedding,
    drop: Dropout,
    h: Vec<GPT2Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl GPT2Model {
    fn new(vocab_size: usize, embed_dim: usize, drop_p: f32, device: &Device) -> Result<Self> {
        let wte_tensor = Tensor::rand(0f32, 1.0, (vocab_size, embed_dim), device)?;
        let wte = Embedding::new(wte_tensor, embed_dim);

        let wpe_tensor = Tensor::rand(0f32, 1.0, (vocab_size, embed_dim), device)?;
        let wpe = Embedding::new(wpe_tensor, embed_dim);

        let drop = Dropout::new(drop_p);

        let ln_tensor = Tensor::rand(0f32, 1.0, (embed_dim,), device)?;
        let ln_bias = Tensor::rand(0f32, 1.0, (embed_dim,), device)?;
        let eps = 1e-05;
        let ln_f = LayerNorm::new(ln_tensor, ln_bias, eps);

        let mut h: Vec<GPT2Block> = Vec::new();
        for _ in 0..12 {
            h.push(GPT2Block::new(embed_dim, device)?);
        }

        let lm_head_tensor = Tensor::rand(0f32, 1.0, (vocab_size, embed_dim), device)?;
        let lm_head = Linear::new(lm_head_tensor, None);

        Ok(GPT2Model {
            wte,
            wpe,
            drop,
            h,
            ln_f,
            lm_head,
        })
    }
    fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let input_emb = self.wte.forward(token_ids)?;
        let position_emb = self.wpe.forward(token_ids)?;

        let mut hidden_states = input_emb.add(&position_emb)?;

        hidden_states = self.drop.forward(&hidden_states, true)?;
        for block in self.h.iter() {
            hidden_states = block.forward(&hidden_states)?;
        }
        hidden_states = self.ln_f.forward(&hidden_states)?;

        let token_logits = self.lm_head.forward(&hidden_states)?;
        Ok(token_logits)
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let model = GPT2Model::new(50275, 768, 0.2, &device)?;

    let token_ids: Vec<u32> = vec![0, 12, 30];
    let token_ids = Tensor::from_vec(token_ids, (1, 3), &device)?;

    let pred = model.forward(&token_ids)?;
    println!("{:?}", pred.shape());
    Ok(())
}
