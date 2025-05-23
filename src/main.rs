use std::{fs::{read_to_string, File}, io::{self, BufRead, BufReader}};

use burn::{backend::{libtorch::LibTorchDevice, LibTorch}, tensor::{Tensor, TensorData}};
use gpt::{model::{self, model::GPTConfig}, run, tokenizer::{CharacterTokenizer, Tokenizer}};

type B = LibTorch<f32, i8>;

fn main() {

    let device = LibTorchDevice::Cpu;

    let tokenizer = CharacterTokenizer::new();

    let model: model::model::GPT<B> = GPTConfig::new(128, tokenizer.vocab_size(), 4, 4, 128, 512).init(&device);

    let prompt = "Hiiii how are you?";

    run::run(&model, &tokenizer, prompt, 500, 64, 42);
    
}
