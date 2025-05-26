use std::io::{self, Write};

use burn::{config::{self, Config}, module::Module, nn::{loss::CrossEntropyLossConfig, Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig}, prelude::Backend, tensor::{activation::softmax, Int, Shape, Tensor, TensorData}};
use rand::{distr::{weighted::WeightedIndex, Distribution}, rngs::StdRng};

use crate::tokenizer::Tokenizer;

use super::decoder::{DecoderBlock, DecoderBlockConfig};

const PADDING_IDX: i32 = 99;


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

    pub fn generate(&self, prompt: &str, ctx_len: usize, n_new_tokens: usize, rng: &mut StdRng, tk: &impl Tokenizer, device: &B::Device) {
        let buf = tk.encode(prompt);
        let mut buf = tk.format(buf.as_slice(), 0);
        for _ in 0..n_new_tokens {
            let x = {
                let idx_slice = &buf[(buf.len() as isize - ctx_len as isize).max(0) as usize..];
                Tensor::<B, 2, Int>::from_data(
                    TensorData::new(idx_slice.to_vec().iter().map(|x| *x as i32).collect()
                    , Shape::new([1, idx_slice.len()])), 
                    &device
                )
            };

            let output = self.forward(x);
            let n = output.dims()[1];
            let slice = output.slice([(0..1), (n-1..n)]).flatten::<1>(0, 2);
            let probs = softmax(slice, 0)
                .into_data()
                .convert::<f32>();
            let prob_slice = probs
                .as_slice::<f32>()
                .expect("Error parsing softmax probabilities");
            let distribution = WeightedIndex::new(
                &prob_slice[..prob_slice.len() - 1]).unwrap();
            let pred = distribution.sample(rng) as usize;
            buf.push(pred);
            let dec = tk.decode(&vec![pred]);
            
            print!("{}", dec);

            if dec == "<E>" {
                break;
            }
            io::stdout().flush().unwrap();
        }
    }

    pub fn loss(&self, logits: Tensor<B, 3>, y: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let [b, t, c] = logits.dims(); // Batch x Time or Token x Channels

        let flat_logits = logits.reshape([b*t, c]);
        let flat_y = y.reshape([b*t]);

        // let mask = flat_y.clone().not_equal_elem(PADDING_IDX);

        // let logit_mask = mask.clone().unsqueeze::<2>().transpose().expand([b*t, c]);

        // let masked_logits = flat_logits.mask_fill(logit_mask, -1e9);

        // let masked_y = flat_y.mask_fill(mask, -1e9);

        CrossEntropyLossConfig::new()   
            .init(&flat_logits.device())
            .forward(flat_logits, flat_y)
    }
}

#[derive(Config)]
pub struct GPTConfig {
    pub ctx_len: usize,
    pub vocab_size: usize,
    pub n_layers: usize, // # of decoder blocks
    pub n_heads: usize,
    pub n_embed: usize, //(d_model)
    pub n_hidden: usize,
    #[config(default = "0.1")]
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
            positional_embedding: EmbeddingConfig::new(self.ctx_len, self.n_embed)
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