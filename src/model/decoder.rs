use burn::{config::{self, Config}, module::Module, nn::{LayerNorm, LayerNormConfig}, prelude::Backend, tensor::Tensor};

use super::{attention::{MultiHeadAttention, MultiHeadAttentionConfig}, positionwise::{PositionWiseFeedForward, PositionWiseFeedForwardConfig}};

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    l_norm1: LayerNorm<B>,
    mha: MultiHeadAttention<B>,
    l_norm2: LayerNorm<B>,
    pwff: PositionWiseFeedForward<B>
}

impl<B: Backend> DecoderBlock<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = input.clone();
        let x = self.mha.forward(self.l_norm1.forward(x), true) + input;
        let x = self.pwff.forward(self.l_norm2.forward(x.clone())) + x;
        x
    }
}

#[derive(Config, Debug)]
pub struct DecoderBlockConfig {
    n_embed: usize, //# of embedding scalars per token (d_model)
    n_hidden: usize,
    n_heads: usize,
    n_layers: usize,
    #[config(default = "0.2")]
    dropout: f64,
}

impl DecoderBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DecoderBlock<B> {
        DecoderBlock { 
            l_norm1: LayerNormConfig::new(self.n_embed).init(device), 
            mha: MultiHeadAttentionConfig::new(self.n_embed, self.n_heads, self.n_layers).init(device), 
            l_norm2: LayerNormConfig::new(self.n_embed).init(device), 
            pwff: PositionWiseFeedForwardConfig::new(self.n_embed, self.n_hidden, self.n_layers).init(device)
        }
    }
}