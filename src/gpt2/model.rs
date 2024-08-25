use core::f32;
use core::ops::Sub;

use candle_core::{IndexOp, Result, Shape, Tensor};
use candle_nn::ops::softmax;
use candle_nn::{
    Activation, Conv1d, Conv1dConfig, Dropout, Embedding, LayerNorm, LayerNormConfig, Linear,
    Module, VarBuilder,
};

pub struct Attention {
    num_heads: usize,
    head_dim: usize,
    c_attn: Conv1d, // Concatenated attention
    c_proj: Conv1d, // Concatenated projection, instead of concating the results of each head, mix
}
impl Attention {
    fn new(num_heads: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let conv_config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let c_attn = candle_nn::conv1d(embed_dim, embed_dim * 3, 1, conv_config, vb.pp("c_attn"))?;
        let c_proj = candle_nn::conv1d(embed_dim, embed_dim, 1, conv_config, vb.pp("c_proj"))?;

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

        let device = kqv.device();

        let inf = Tensor::full(f32::MAX, (num_tokens, num_tokens), device)?;
        let tril = Tensor::tril2(num_tokens, candle_core::DType::F32, device)?.sub(1.0)?;
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
    pub train: bool,
    c_fc: Conv1d,
    c_proj: Conv1d,
    c_act: Activation,
    c_dropout: Dropout,
}
impl GPT2MLP {
    pub fn new(intermediate_size: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let conv_config = Conv1dConfig {
            padding: 0,
            dilation: 1,
            groups: 1,
            stride: 1,
        };
        let c_fc = candle_nn::conv1d(embed_dim, intermediate_size, 1, conv_config, vb.pp("c_fc"))?;
        let c_proj = candle_nn::conv1d(
            intermediate_size,
            embed_dim,
            1,
            conv_config,
            vb.pp("c_proj"),
        )?;

        let c_act = Activation::Gelu;
        let c_dropout = Dropout::new(0.2);
        let train = false;

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
        let hidden_states = self.c_fc.forward(&hidden_states.transpose(1, 2)?)?;
        let hidden_states = hidden_states.transpose(1, 2)?;
        let hidden_states = self.c_act.forward(&hidden_states)?;
        let hidden_states = self.c_proj.forward(&hidden_states.transpose(1, 2)?)?;
        let hidden_states = hidden_states.transpose(1, 2)?;
        let hidden_states = self.c_dropout.forward(&hidden_states, self.train)?;
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
    fn new(embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let ln_c = LayerNormConfig {
            eps: 1e-05,
            remove_mean: true,
            affine: true,
        };
        let ln_1 = candle_nn::layer_norm(embed_dim, ln_c, vb.pp("layer_norm_1"))?;

        let num_heads = 12;
        let attn = Attention::new(num_heads, embed_dim, vb.pp("attn"))?;

        let ln_2 = candle_nn::layer_norm(embed_dim, ln_c, vb.pp("layer_norm_2"))?;

        let mlp = GPT2MLP::new(4 * embed_dim, embed_dim, vb.pp("mlp"))?;

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
    pub train: bool,
    pub wte: Embedding,
    pub wpe: Embedding,
    pub drop: Dropout,
    pub h: Vec<GPT2Block>,
    pub ln_f: LayerNorm,
    pub lm_head: Linear,
}

impl GPT2Model {
    pub fn new(vocab_size: usize, embed_dim: usize, drop_p: f32, vb: VarBuilder) -> Result<Self> {
        let wte = candle_nn::embedding(vocab_size, embed_dim, vb.pp("wte"))?;
        let wpe = candle_nn::embedding(vocab_size, embed_dim, vb.pp("wpe"))?;
        let drop = Dropout::new(drop_p);

        let ln_config = LayerNormConfig {
            eps: 1e-05,
            remove_mean: true,
            affine: true,
        };
        let ln_f = candle_nn::layer_norm(embed_dim, ln_config, vb.pp("layer_norm_f"))?;

        let mut h: Vec<GPT2Block> = Vec::new();
        for block_num in 0..12 {
            h.push(GPT2Block::new(
                embed_dim,
                vb.pp(format!("block_{block_num}")),
            )?);
        }

        let lm_head = candle_nn::linear(embed_dim, vocab_size, vb.pp("lm_head"))?;
        let train = false;

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
        let position_emb = self.wpe.forward(token_ids)?;

        let mut hidden_states = input_emb.add(&position_emb)?;

        hidden_states = self.drop.forward(&hidden_states, self.train)?;
        for block in self.h.iter() {
            hidden_states = block.forward(&hidden_states)?;
        }
        hidden_states = self.ln_f.forward(&hidden_states)?;

        let token_logits = self.lm_head.forward(&hidden_states)?;
        Ok(token_logits)
    }
}
