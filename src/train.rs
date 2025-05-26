use std::{process::exit, time};

use burn::{config::Config, module::{AutodiffModule, Module}, optim::{AdamWConfig, GradientsParams, Optimizer}, prelude::Backend, record::{CompactRecorder, NamedMpkFileRecorder}, tensor::{backend::AutodiffBackend, ElementConversion, Int, Tensor, TensorData}};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{model::model::{GPTConfig, GPT}, tokenizer::{CharacterTokenizer, Tokenizer}};

const SAMPLE_PROMPT: &str = "BIANCA:
If you affect him, sister, here I swear
I'll plead for you myself, but you shall have
him.";

#[derive(Config)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub batches_per_step: usize,
    pub seed: u64,
    pub model: GPTConfig,
    pub optim: AdamWConfig
}

pub fn train<B: AutodiffBackend>(
    config: &TrainingConfig, 
    train_data: Vec<usize>,
    valid_data: Vec<usize>,
    continue_progress: bool,
    tk: &impl Tokenizer,
    device: B::Device,
) {
    let mut rng = StdRng::seed_from_u64(config.seed);

    let recorder = CompactRecorder::new();

    B::seed(config.seed);

    let mut model: GPT<B> = config.model.init(&device);

    if continue_progress {
        println!("Loading model...");
        model = model.load_file("sm-tmp/model-4", &recorder, &device).expect("Error loading model");
        println!("Model Loaded!");
    }

    model.clone()
        .save_file("sm-tmp/model-inital", &recorder)
        .expect("Error saving initial model");

    let mut optim: burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, GPT<B>, B> = config.optim.init::<B, GPT<B>>();

    for epoch in 1..config.epochs + 1 {

        let start = time::Instant::now();

        let valid_loss = {
            let [x, y] = batch(&mut rng, valid_data.clone(), config.batch_size, config.model.ctx_len, tk, &device);

            let v_model = model.clone().valid();
            let output = v_model.forward(x);

            // println!("{}", output);
            println!("SAMPLE OUTPUT: ");
            
            v_model.generate(SAMPLE_PROMPT, config.model.ctx_len, 200, &mut rng, tk, &device);
            
            println!("\n----------------------------------------------");

            let loss = v_model.loss(output, y);
            loss.into_scalar().elem::<f64>()
            
        };

        let mut train_loss = 0.0;

        for s in 1..=config.batches_per_step {
            let [x, y] = batch(&mut rng, train_data.clone(), config.batch_size, config.model.ctx_len, tk, &device);

            let output = model.forward(x);
            let loss = model.loss(output, y);
            train_loss += loss.clone().into_scalar().elem::<f64>();

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.lr, model, grads);
        } 

        let elapsed = start.elapsed().as_secs();

        let epochs_left = (config.epochs - epoch) as u64;

        let est_s = elapsed * epochs_left;

        // let h = (elapsed / 60) / 60;
        let m = elapsed / 60;

        println!("Epoch: {} / {} || Training Loss: {} || Validation Loss: {} || Time Taken: {}m {}s  || Time Left: {}H {}m {}s\n----------------------------------------------",
            epoch, 
            config.epochs, 
            train_loss / config.batches_per_step as f64, 
            valid_loss, 
            m,
            elapsed % 60, 
            ((est_s / 60) / 60),
            (est_s / 60) % 60,
            est_s % 60);
        
        println!("Saving model...");

        model.clone().save_file(format!("sm-tmp/model-{}", epoch % 20), &recorder).expect("Error saving model");
        
        println!("Model Saved! - {}", format!("sm-tmp/model-{}", epoch % 20));
    }
    println!("TRAINING COMPLETE!")

}

pub fn batch<B: Backend>(
    rng: &mut impl Rng,
    data: Vec<usize>,
    batch_size: usize,
    ctx_len: usize,
    tk: &impl Tokenizer, 
    device: &B::Device
) -> [Tensor<B, 2, Int>; 2] {

    // let sample_len = rng.random_range(ctx_len / 2..ctx_len - 1);

    let sample_len = ctx_len - 2;

    //get random parts of the data
    let idx = 
        (0..batch_size)
        .map(|_| rng.random_range(0..data.len() - ctx_len - 1))
        .collect::<Vec<usize>>();

    let x_slice: Vec<&[usize]> = idx
            .iter()
            .map(|&i| &data[i..i + sample_len])
            .collect();

    let y_slice: Vec<&[usize]> = idx
        .iter()
        .map(|&i| &data[i + 1..i + 1 + sample_len])
        .collect();

    let mut x_sample = Vec::new();
    let mut y_sample = Vec::new();

    for s in x_slice {
        x_sample.extend(tk.format(s, ctx_len));
    }

    for s in y_slice {
        y_sample.extend(tk.format(s, ctx_len));
    }

    let x_sample: Vec<i32> = x_sample
        .iter()
        .map(|e| *e as i32)
        .collect();

    let y_sample: Vec<i32> = y_sample
        .iter()
        .map(|e| *e as i32)
        .collect();

    //get x samples
    let x = 
        Tensor::from_data(TensorData::new(x_sample, [batch_size, ctx_len]), device);

    //get y samples (x + 1)
    let y = 
        Tensor::from_data(TensorData::new(y_sample, [batch_size, ctx_len]), device);

    [x, y]
}