use core::f32;
use core::ops::{Div, Sub};

use candle_core::{IndexOp, Result, Shape, Tensor, D};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{
    Activation, Conv1d, Conv1dConfig, Dropout, Embedding, LayerNorm, LayerNormConfig, Linear,
    Module, VarBuilder,
};

pub fn conv1d_kernel1(out_channels: usize, in_channels: usize, vb: VarBuilder) -> Result<Conv1d> {
    let cfg = Conv1dConfig::default();
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((in_channels / cfg.groups, out_channels), "weight", init_ws)?;
    let ws = ws.transpose(0, 1)?;
    let ws = ws.unsqueeze(D::Minus1)?; // They had a custom
    let bound = 1. / (in_channels as f64).sqrt();
    let init_bs = candle_nn::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_channels, "bias", init_bs)?;
    Ok(Conv1d::new(ws, Some(bs), cfg))
}

pub struct Attention {
    num_heads: usize,
    head_dim: usize,
    c_attn: Conv1d, // Concatenated attention
    c_proj: Conv1d, // Concatenated projection, instead of concating the results of each head, mix
}
impl Attention {
    fn new(num_heads: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let c_attn = conv1d_kernel1(embed_dim * 3, embed_dim, vb.pp("c_attn"))?;
        let c_proj = conv1d_kernel1(embed_dim, embed_dim, vb.pp("c_proj"))?;

        let head_dim = embed_dim / num_heads;
        Ok(Self {
            num_heads,
            head_dim,
            c_attn,
            c_proj,
        })
    }
}
impl Module for Attention {
    fn forward(&self, hidden_state: &Tensor) -> Result<Tensor> {
        let (batch_size, num_tokens, embed_dim) = hidden_state.dims3()?;
        // let hidden_state = hidden_state.transpose(1, 2)?;
        let mut kqv = self.c_attn.forward(&hidden_state.transpose(1, 2)?)?;
        kqv = kqv.transpose(1, 2)?; // Shape: B, T, embed_dim * 3
        kqv = kqv.reshape((batch_size, num_tokens, 3, embed_dim))?;

        let mut q = kqv.i((.., .., 0, ..))?; // batch_size, num_tokens, embed_dim
        let mut k = kqv.i((.., .., 1, ..))?;
        let mut v = kqv.i((.., .., 2, ..))?;

        let org_shape = q.shape().clone();
        let new_shape = Shape::from_dims(&[batch_size, num_tokens, self.num_heads, self.head_dim]);

        q = q.reshape(&new_shape)?;
        k = k.reshape(&new_shape)?;
        v = v.reshape(&new_shape)?;

        q = q.permute((0, 2, 1, 3))?.contiguous()?; // b, heads, seq_length, head_features
        k = k.permute((0, 2, 1, 3))?.contiguous()?;
        v = v.permute((0, 2, 1, 3))?.contiguous()?;

        let device = kqv.device();

        // Create the lower triangular mask using `tril2`
        let mut causal_mask = Tensor::tril2(num_tokens, candle_core::DType::F32, device)?; // Shape: (seq_len, seq_len)
        let mask_value = 1e9;
        causal_mask = (causal_mask.sub(1.0)? * mask_value)?; // Converts ones to zeros and zeros to `-1e9`
                                                             // println!("Causal Mask is {causal_mask}");

        // lhs 8, 12, 4, 64
        let mut weights = q.matmul(&k.transpose(2, 3)?)?;
        weights = weights.div((self.head_dim as f64).sqrt())?;
        weights = weights.broadcast_add(&causal_mask)?; // do not let previous tokens see the future
        weights = softmax_last_dim(&weights)?;

        let dropout = Dropout::new(0.2);
        weights = dropout.forward(&weights, true)?;

        // println!("Weights {weights}"); // Thsese look good!
        // TODO add dropout to the weights!
        let mut attn = weights.matmul(&v)?; // (B, heads, seq_length, head_features)
        attn = attn.reshape(org_shape)?;
        attn = self.c_proj.forward(&attn.transpose(1, 2)?)?;
        attn = attn.transpose(1, 2)?;
        Ok(attn)
    }
}

pub struct GPT2MLP {
    pub train: bool,
    c_fc: Conv1d,
    c_proj: Conv1d,
    c_act: Activation,
    c_dropout: Dropout,
}
impl GPT2MLP {
    pub fn new(intermediate_size: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let c_fc = conv1d_kernel1(intermediate_size, embed_dim, vb.pp("c_fc"))?;
        let c_proj = conv1d_kernel1(embed_dim, intermediate_size, vb.pp("c_proj"))?;

        let c_act = Activation::Gelu;
        let c_dropout = Dropout::new(0.2);
        let train = true;

        Ok(GPT2MLP {
            train,
            c_fc,
            c_proj,
            c_act,
            c_dropout,
        })
    }
}
impl Module for GPT2MLP {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.c_fc.forward(&hidden_states.transpose(1, 2)?)?;
        hidden_states = hidden_states.transpose(1, 2)?;
        hidden_states = self.c_act.forward(&hidden_states)?;
        hidden_states = self.c_proj.forward(&hidden_states.transpose(1, 2)?)?;
        hidden_states = hidden_states.transpose(1, 2)?;
        hidden_states = self.c_dropout.forward(&hidden_states, self.train)?;
        Ok(hidden_states)
    }
}

pub struct GPT2Block {
    ln_1: LayerNorm,
    attn: Attention,
    drop: Dropout,
    ln_2: LayerNorm,
    mlp: GPT2MLP,
}
impl GPT2Block {
    fn new(embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let ln_c = LayerNormConfig {
            eps: 1e-05,
            remove_mean: false,
            affine: false,
        };
        let ln_1 = candle_nn::layer_norm(embed_dim, ln_c, vb.pp("ln_1"))?;

        let num_heads = 12;
        let attn = Attention::new(num_heads, embed_dim, vb.pp("attn"))?;

        let ln_2 = candle_nn::layer_norm(embed_dim, ln_c, vb.pp("ln_2"))?;

        let mlp = GPT2MLP::new(4 * embed_dim, embed_dim, vb.pp("mlp"))?;

        let drop = Dropout::new(0.2);

        Ok(Self {
            ln_1,
            attn,
            drop,
            ln_2,
            mlp,
        })
    }
}

impl Module for GPT2Block {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();

        let mut hidden_states = self.ln_1.forward(hidden_states)?;

        let attn = self.attn.forward(&hidden_states)?;
        let attn = self.drop.forward(&attn, true)?; // Assuming `self.dropout` is defined

        hidden_states = residual.add(&attn)?;

        let residual = hidden_states.clone();
        hidden_states = self.ln_2.forward(&hidden_states)?;

        let ff_hidden_states = self.mlp.forward(&hidden_states)?;
        hidden_states = residual.add(&ff_hidden_states)?;

        Ok(hidden_states)
    }
}

pub struct GPT2Model {
    pub train: bool,
    pub wte: Embedding,
    pub wpe: Embedding,
    pub drop: Dropout,
    pub h: Vec<GPT2Block>,
    pub ln_f: LayerNorm,
    pub lm_head: Linear,
}

impl GPT2Model {
    pub fn new(
        vocab_size: usize,
        context_length: usize,
        embed_dim: usize,
        drop_p: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let wte = candle_nn::embedding(vocab_size, embed_dim, vb.pp("wte"))?;
        let wpe = candle_nn::embedding(context_length, embed_dim, vb.pp("wpe"))?;
        let drop = Dropout::new(drop_p);

        let ln_config = LayerNormConfig {
            eps: 1e-05,
            remove_mean: false,
            affine: false,
        };
        let ln_f = candle_nn::layer_norm(embed_dim, ln_config, vb.pp("ln_f"))?;

        let mut h: Vec<GPT2Block> = Vec::new();
        for block_num in 0..12 {
            h.push(GPT2Block::new(embed_dim, vb.pp(format!("h.{block_num}")))?);
        }

        let lm_head = candle_nn::linear_no_bias(embed_dim, vocab_size, vb.pp("lm_head"))?;
        let train = true;

        Ok(GPT2Model {
            train,
            wte,
            wpe,
            drop,
            h,
            ln_f,
            lm_head,
        })
    }
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let input_emb = self.wte.forward(token_ids)?;
        let (_, seq_length) = token_ids.dims2()?;
        let positions = Tensor::arange(0 as u32, seq_length as u32, token_ids.device())?;
        let position_emb = self.wpe.forward(&positions)?;
        let mut hidden_states = input_emb.broadcast_add(&position_emb)?;

        hidden_states = self.drop.forward(&hidden_states, self.train)?;
        for block in self.h.iter() {
            hidden_states = block.forward(&hidden_states)?;
        }
        // println!("FInished processing blocks");
        hidden_states = self.ln_f.forward(&hidden_states)?;

        // println!("FInished ln f");
        let token_logits = self.lm_head.forward(&hidden_states)?;
        // println!("{token_logits}");
        Ok(token_logits)
    }
}
