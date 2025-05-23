use burn::{config::{self, Config}, module::Module, nn::{loss::CrossEntropyLossConfig, Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig}, prelude::Backend, tensor::{Int, Tensor}};

use super::decoder::{DecoderBlock, DecoderBlockConfig};

#[derive(Module, Debug)]
pub struct GPT<B: Backend> {
    token_embedding: Embedding<B>,
    positional_embedding: Embedding<B>,
    decoder_blocks: Vec<DecoderBlock<B>>,
    dropout: Dropout,
    l_norm: LayerNorm<B>,
    linear: Linear<B>,
}

impl<B: Backend> GPT<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.token_embedding.forward(input.clone());

        let x = {
            let [_, t] = input.dims(); // B x T
            let pos_embed = self.positional_embedding.forward(
                Tensor::arange(0..(t as i64), &x.device()).unsqueeze()
            );
            x + pos_embed
        };

        let x = self.dropout.forward(x);
        let x = self.decoder_blocks
            .iter()
            .fold(x, |x, block| {
                block.forward(x)
            }
        );
        let x = self.l_norm.forward(x);
        let x = self.linear.forward(x);
        x
    }

    pub fn loss(&self, logits: Tensor<B, 3>, y: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let [b, t, c] = logits.dims(); // Batch x Time or Token x Channels
        CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.reshape([b*t, c]), y.reshape([b*t]))
    }
}

#[derive(Config)]
pub struct GPTConfig {
    pub context_length: usize,
    pub vocab_size: usize,
    pub n_layers: usize, // # of decoder blocks
    pub n_heads: usize,
    pub n_embed: usize, //(d_model)
    pub n_hidden: usize,
    #[config(default = "0.2")]
    pub dropout: f64
}

impl GPTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GPT<B> {
        GPT {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.n_embed)
                .with_initializer(burn::nn::Initializer::Normal { 
                    mean: 0.0, 
                    std: 0.02 }
                ).init(device),
            positional_embedding: EmbeddingConfig::new(self.context_length, self.n_embed)
                .with_initializer(burn::nn::Initializer::Normal { 
                    mean: 0.0, 
                    std: 0.02 }
                ).init(device),
            decoder_blocks: (0..self.n_layers).map(|_| {
                DecoderBlockConfig::new(self.n_embed, self.n_hidden, self.n_heads, self.n_layers).init(device)
            }).collect(),
            dropout: DropoutConfig::new(self.dropout).init(),
            l_norm: LayerNormConfig::new(self.n_embed).init(device),
            linear: LinearConfig::new(self.n_embed, self.vocab_size)
                .with_initializer(burn::nn::Initializer::Normal { 
                    mean: 0.0, 
                    std: 0.02
                 }
                ).init(device),
        }
    }
}