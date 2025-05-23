use std::process::exit;

use burn::{config::{self, Config}, module::Module, nn::{Dropout, DropoutConfig, Linear, LinearConfig}, prelude::Backend, tensor::{activation::softmax, Bool, Tensor}};

// pub fn scaled_dot_product<B: Backend>(  
//     q: Tensor<B, 4>, 
//     k: Tensor<B, 4>, 
//     v: Tensor<B, 4>, 
//     apply_mask: bool) 
//         -> [Tensor<B, 4>; 2] {
//     let shape = q.shape(); //Shape: batch x n_heads x sequence length? x head_dim
//     let d_k = shape.dims.last().expect("Error reading d_k");
//     let mut scaled = q.matmul(k.transpose()).div_scalar((*d_k as f32).sqrt());
//     if apply_mask {
//         let mask = Tensor::<B, 2, Bool>::tril_mask([seq_le, seq_le], 0, &input.device());
//         scaled.mask_fill(mask.unsqueeze(), f32::NEG_INFINITY);
//     }
//     let attention = softmax(scaled, 3);
//     let values = attention.clone().matmul(v);
//     [values, attention]
// }

#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    n_embed: usize,
    n_heads: usize,
    n_layers: usize,
    #[config(default = "0.2")]
    dropout: f64,
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    n_embed: usize, //# of embedding scalars (d_model)
    n_heads: usize, // # of self attention blocks
    head_dim: usize, // n_embed / n_heads
    attn_dropout: Dropout,
    resid_dropout: Dropout,
    qkv: Linear<B>,
    l_out: Linear<B>,
}

impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        let head_dim = self.n_embed / self.n_heads; //d_k
        assert!(head_dim * self.n_heads == self.n_embed);

        MultiHeadAttention { 
            n_embed: self.n_embed,
            n_heads: self.n_heads,
            head_dim: head_dim,
            attn_dropout: DropoutConfig::new(self.dropout).init(),
            resid_dropout: DropoutConfig::new(self.dropout).init(),
            qkv: LinearConfig::new(self.n_embed, 3 * self.n_embed).init(device),
            l_out:  LinearConfig::new(self.n_embed, self.n_embed).init(device)
        }
    }
}

impl<B: Backend> MultiHeadAttention<B> {
    // pub fn new(device: &<B as Backend>::Device, n_embed: usize, n_heads: usize) -> Self {
    //     let head_dim = n_embed/n_heads;
            
    //     assert_eq!(head_dim*n_heads, n_embed, "ERROR -- Multi Head Attention: # of embeddings must be divisible by # of heads");
        
    //     Self {
    //         n_embed,
    //         n_heads,
    //         head_dim,
    //         attn_dropout: Drop
    //         qkv: LinearConfig::new(n_embed, 3 * n_embed).init(device),
    //         l_out: LinearConfig::new(n_embed, n_embed).init(device),
    //     }
    // }

    pub fn forward(&self, input: Tensor<B, 3>, apply_mask: bool) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = input.shape().dims(); // B x T x C
        
        let qkv = self.qkv.forward(input.clone()); // batch size x seq_len x 3 * n_embed

        let qkv = qkv.reshape([batch_size, seq_len, self.n_heads, 3 * self.head_dim]);
        let qkv = qkv.permute([0, 2, 1, 3]);

        let qkv = qkv.chunk(3, 3);

        assert_eq!(qkv.len(), 3, "Error -- Multi Head Attention: QKV vec is not of length 3");

        let [q, k, v] = 
            [qkv[0].clone(), qkv[1].clone(), qkv[2].clone()];

        let shape = q.shape(); //Shape: batch x n_heads x sequence length? x head_dim

        let d_k = shape.dims.last().expect("Error reading d_k");

        let mut scaled = q.matmul(k.transpose()).div_scalar((*d_k as f32).sqrt());

        if apply_mask {
            let mask = Tensor::<B, 2, Bool>::tril_mask([seq_len, seq_len], 0, &input.device());
            scaled = scaled.mask_fill(mask.unsqueeze(), f32::NEG_INFINITY);
        }
        
        let attention = softmax(scaled, 3);
        let x = self.attn_dropout.forward(attention);
        let x = x.clone().matmul(v); //values
        let x = x.reshape([batch_size, seq_len, self.n_heads * self.head_dim]);
        let x = self.resid_dropout.forward(x);
        self.l_out.forward(x)
    }
}