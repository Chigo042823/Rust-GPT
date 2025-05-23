use std::io::{self, Write};

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
    tokenizer: &impl Tokenizer, 
    prompt: &str, 
    n_new_tokens: usize, 
    ctx_len: usize,
    seed: u64) {
        let device = <B as Backend>::Device::default();
        let mut rng = StdRng::seed_from_u64(seed);

        let mut indecies = tokenizer.encode(&prompt);

        println!("PROMPT: {prompt}");

        for _ in 0..n_new_tokens {
            let x = {
                let idx_slice = &indecies[(indecies.len() as isize - ctx_len as isize).max(0) as usize..];
                Tensor::<B, 2, Int>::from_data(
                    TensorData::new(idx_slice.to_vec().iter().map(|x| *x as i32).collect()
                    , Shape::new([1, idx_slice.len()])), 
                    &device
                )
            };

            let output = model.forward(x);
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
            let pred = distribution.sample(&mut rng) as usize;
            indecies.push(pred);
            let dec = tokenizer.decode(&vec![pred]);
            
            print!("{}", dec);

            if dec == "<E>" {
                return;
            }
            io::stdout().flush().unwrap();
        }
        println!();

    }