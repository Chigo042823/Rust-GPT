use std::io::{self, stdin, stdout, BufRead, Write};

use burn::tensor::activation::softmax;
use burn::tensor::{Int, Shape, Tensor, TensorData};
use burn::{prelude::*, tensor::Device};

use rand::distr::Distribution;

use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::tokenizer::{self, Tokenizer};

use crate::model::model::GPT;

pub fn run<B:Backend>(
    model: &GPT<B>, 
    tk: &impl Tokenizer,
    n_new_tokens: usize, 
    ctx_len: usize,
    seed: u64) {
        let device = <B as Backend>::Device::default();
        let mut rng = StdRng::seed_from_u64(seed);

        loop {

        print!("PROMPT: ");

        let _ = io::stdout().flush();

        let mut prompt = String::new();

        let _ = io::stdin().read_line(&mut prompt);

        model.generate(&prompt, ctx_len, n_new_tokens, &mut rng, tk, &device);

        println!();
        }
    }