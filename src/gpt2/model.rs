use core::f32;

use candle_core::{IndexOp, Result, Shape, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{Activation, Dropout, Embedding, LayerNorm, LayerNormConfig, Module, VarBuilder};

struct GPT2Conv1D {
    num_output_features: usize,
    weight: Tensor,
    bias: Tensor,
}
impl GPT2Conv1D {
    fn new(nf: usize, nx: usize, vb: VarBuilder) -> Result<Self> {
        let init_weight = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let weight = vb.get_with_hints((nx, nf), "weight", init_weight)?;
        let init_bias = candle_nn::init::Init::Const(0.0);
        let bias = vb.get_with_hints((nf,), "bias", init_bias)?;
        Ok(GPT2Conv1D {
            weight,
            bias,
            num_output_features: nf,
        })
    }
}

impl Module for GPT2Conv1D {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, num_tokens, embed_dim) = x.dims3()?;
        let output_shape = Shape::from_dims(&[batch_size, num_tokens, self.num_output_features]);

        let mut output = x
            .reshape(&[batch_size * num_tokens, embed_dim])?
            .matmul(&self.weight)?;

        output = output.broadcast_add(&self.bias)?;
        output = output.reshape(output_shape)?;
        Ok(output)
    }
}

// #[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub embd_pdrop: f32,
    pub num_hidden_layers: usize, // Number of blocks to use
    pub layer_norm_epsilon: f64,
    // self._attn_implementation = config._attn_implementation
}
impl Default for Config {
    fn default() -> Config {
        Config {
            hidden_size: 768,
            max_position_embeddings: 1024,
            embd_pdrop: 0.2,
            vocab_size: 50257,
            num_hidden_layers: 12,
            layer_norm_epsilon: 1e-5,
        }
    }
}

pub struct Attention {
    num_heads: usize,
    head_dim: usize,
    c_attn: GPT2Conv1D, // Concatenated attention
    c_proj: GPT2Conv1D, // Concatenated projection, instead of concating the results of each head, mix
    train_mode: bool,
    drop: Dropout,
    bias: Tensor,
}
impl Attention {
    fn new(num_heads: usize, embed_dim: usize, drop_p: f32, vb: VarBuilder) -> Result<Self> {
        let c_attn = GPT2Conv1D::new(embed_dim * 3, embed_dim, vb.pp("c_attn"))?;
        let c_proj = GPT2Conv1D::new(embed_dim, embed_dim, vb.pp("c_proj"))?;

        let head_dim = embed_dim / num_heads;
        let bias = vb.get((1, 1, 1024, 1024), "bias")?;

        Ok(Self {
            num_heads,
            head_dim,
            c_attn,
            c_proj,
            train_mode: false,
            drop: Dropout::new(drop_p),
            bias,
        })
    }
    fn train(&mut self) {
        self.train_mode = true;
    }
    fn eval(&mut self) {
        self.train_mode = false;
    }
}
impl Module for Attention {
    fn forward(&self, hidden_state: &Tensor) -> Result<Tensor> {
        let (batch_size, num_tokens, embed_dim) = hidden_state.dims3()?;
        let mut qkv = self.c_attn.forward(&hidden_state)?;

        qkv = qkv.reshape((batch_size, num_tokens, 3, embed_dim))?;

        let mut q = qkv.i((.., .., 0, ..))?; // batch_size, num_tokens, embed_dim
        let mut k = qkv.i((.., .., 1, ..))?;
        let mut v = qkv.i((.., .., 2, ..))?;

        let org_shape = q.shape().clone();
        let new_shape = Shape::from_dims(&[batch_size, num_tokens, self.num_heads, self.head_dim]);

        q = q.reshape(&new_shape)?;
        k = k.reshape(&new_shape)?;
        v = v.reshape(&new_shape)?;

        q = q.permute((0, 2, 1, 3))?.contiguous()?; // b, heads, seq_length, head_features
        k = k.permute((0, 2, 1, 3))?.contiguous()?;
        v = v.permute((0, 2, 1, 3))?.contiguous()?;

        // Query Key and Value are perfect, don't need to check here..
        // println!("q {}", q.mean_all()?);
        // println!("k {}", k.mean_all()?);
        // println!("v {}", v.mean_all()?);

        // Creating causal_mask
        let mut causal_mask = self.bias.i((.., .., 0..num_tokens, 0..num_tokens))?;
        causal_mask = (causal_mask.neg()? + 1.0)?;
        causal_mask = (causal_mask * std::f32::MIN as f64)?.to_dtype(candle_core::DType::F32)?; // Be wary of numerical overflow here...TODO

        // LHS: (B, H, T, head_features),
        let mut attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let scale_factor = 1.0 / (self.head_dim as f64).sqrt(); // Hand checked this is good
        attn_weights = (attn_weights * scale_factor)?;

        attn_weights = attn_weights.broadcast_add(&causal_mask)?;
        attn_weights = softmax_last_dim(&attn_weights)?;
        attn_weights = self.drop.forward(&attn_weights, self.train_mode)?;

        let mut attn = attn_weights.matmul(&v)?; // (batch, heads, seq_length, head_features)
        attn = attn.transpose(1, 2)?.contiguous()?;
        attn = attn.reshape(org_shape)?; // (batch, seq_length, embed_dim)
        attn = self.c_proj.forward(&attn)?;
        Ok(attn)
    }
}

pub struct GPT2MLP {
    train_mode: bool,
    c_fc: GPT2Conv1D,
    c_proj: GPT2Conv1D,
    c_act: Activation,
    c_dropout: Dropout,
}
impl GPT2MLP {
    pub fn new(intermediate_size: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let c_fc = GPT2Conv1D::new(intermediate_size, embed_dim, vb.pp("c_fc"))?;
        let c_proj = GPT2Conv1D::new(embed_dim, intermediate_size, vb.pp("c_proj"))?;

        let c_act = Activation::Gelu;
        let c_dropout = Dropout::new(0.2);
        let train = false;

        Ok(GPT2MLP {
            train_mode: train,
            c_fc,
            c_proj,
            c_act,
            c_dropout,
        })
    }
    fn train(&mut self) {
        self.train_mode = true;
    }
    fn eval(&mut self) {
        self.train_mode = false;
    }
}
impl Module for GPT2MLP {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.c_fc.forward(&hidden_states)?;
        hidden_states = self.c_act.forward(&hidden_states)?;
        hidden_states = self.c_proj.forward(&hidden_states)?;
        hidden_states = self.c_dropout.forward(&hidden_states, self.train_mode)?;
        Ok(hidden_states)
    }
}

pub struct GPT2Block {
    ln_1: LayerNorm,
    attn: Attention,
    drop: Dropout,
    ln_2: LayerNorm,
    mlp: GPT2MLP,
    train_mode: bool,
}
impl GPT2Block {
    fn new(embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let ln_c = LayerNormConfig {
            eps: 1e-05,
            remove_mean: true,
            affine: true,
        };
        let ln_1 = candle_nn::layer_norm(embed_dim, ln_c, vb.pp("ln_1"))?;

        let num_heads = 12;
        let attn = Attention::new(num_heads, embed_dim, 0.2, vb.pp("attn"))?;

        let ln_2 = candle_nn::layer_norm(embed_dim, ln_c, vb.pp("ln_2"))?;

        let mlp = GPT2MLP::new(4 * embed_dim, embed_dim, vb.pp("mlp"))?;

        let drop = Dropout::new(0.2);

        Ok(Self {
            ln_1,
            attn,
            drop,
            ln_2,
            mlp,
            train_mode: false,
        })
    }
    fn train(&mut self) {
        self.train_mode = true;
        self.attn.train();
        self.mlp.train();
    }
    fn eval(&mut self) {
        self.train_mode = false;
        self.attn.eval();
        self.mlp.eval();
    }
}

impl Module for GPT2Block {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();

        let mut hidden_states = self.ln_1.forward(hidden_states)?;
        let attn = self.attn.forward(&hidden_states)?;
        // println!("hidden {}", hidden_states.mean_all()?);

        let attn = self.drop.forward(&attn, self.train_mode)?;
        hidden_states = residual.add(&attn)?;

        let residual = hidden_states.clone();
        hidden_states = self.ln_2.forward(&hidden_states)?;

        let ff_hidden_states = self.mlp.forward(&hidden_states)?;
        hidden_states = residual.add(&ff_hidden_states)?;

        Ok(hidden_states)
    }
}

pub struct GPT2Model {
    pub train_mode: bool,
    pub wte: Embedding,
    pub wpe: Embedding,
    pub drop: Dropout,
    pub h: Vec<GPT2Block>,
    pub ln_f: LayerNorm,
}

impl GPT2Model {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let config = Config::default();
        let wte = candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("wte"))?;
        let wpe = candle_nn::embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("wpe"),
        )?;
        let drop = Dropout::new(config.embd_pdrop);

        let ln_config = LayerNormConfig {
            eps: config.layer_norm_epsilon,
            remove_mean: true,
            affine: true,
        };
        let ln_f = candle_nn::layer_norm(config.hidden_size, ln_config, vb.pp("ln_f"))?;

        let mut h: Vec<GPT2Block> = Vec::new();
        for block_num in 0..12 {
            h.push(GPT2Block::new(
                config.hidden_size,
                vb.pp(format!("h.{block_num}")),
            )?);
        }

        let train = false;

        Ok(GPT2Model {
            train_mode: train,
            wte,
            wpe,
            drop,
            h,
            ln_f,
        })
    }
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let input_emb = self.wte.forward(token_ids)?;
        let (_, seq_length) = token_ids.dims2()?;
        let positions = Tensor::arange(0 as u32, seq_length as u32, token_ids.device())?;
        let position_emb = self.wpe.forward(&positions)?;
        let mut hidden_states = input_emb.broadcast_add(&position_emb)?;

        hidden_states = self.drop.forward(&hidden_states, self.train_mode)?;
        for block in self.h.iter() {
            hidden_states = block.forward(&hidden_states)?;
        }
        hidden_states = self.ln_f.forward(&hidden_states)?;

        let token_logits =
            hidden_states.broadcast_matmul(&self.wte.embeddings().transpose(0, 1)?)?;

        Ok(token_logits)
    }

    pub fn train(&mut self) {
        self.train_mode = true;
        for h in self.h.iter_mut() {
            h.train();
        }
    }

    pub fn eval(&mut self) {
        self.train_mode = false;
        for h in self.h.iter_mut() {
            h.eval();
        }
    }
}
