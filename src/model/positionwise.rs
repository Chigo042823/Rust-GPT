use burn::{config::{self, Config}, module::Module, nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig}, prelude::Backend, tensor::Tensor};

#[derive(Config)]
pub struct PositionWiseFeedForwardConfig {
    pub n_embed: usize,
    pub n_hidden: usize,
    pub n_layer: usize, //residual layers
    #[config(default = 0.2)]
    pub dropout: f64
}

#[derive(Module, Debug)]
pub struct PositionWiseFeedForward<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    gelu: Gelu,
    dropout: Dropout
}

impl PositionWiseFeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PositionWiseFeedForward<B> {
        PositionWiseFeedForward { 
            l1: LinearConfig::new(self.n_embed, self.n_hidden)
                .with_initializer(
                    burn::nn::Initializer::Normal { 
                        mean: 0.0, 
                        std: 0.02 
                    })
                    .init(device), 
            l2: LinearConfig::new(self.n_hidden, self.n_embed)
                .with_initializer(
                    burn::nn::Initializer::Normal { 
                        mean: 0.0, 
                        std: 0.02 / (2.0 * self.n_layer as f64).sqrt(),
                    })
                    .init(device), 
            gelu: Gelu::new(), 
            dropout: DropoutConfig::new(self.dropout).init()
        }
    }
}

impl<B: Backend> PositionWiseFeedForward<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.l1.forward(input);
        let x = self.gelu.forward(x);
        let x = self.l2.forward(x);
        let x = self.dropout.forward(x);
        x
    }
}