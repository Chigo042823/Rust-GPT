use std::{fs::File, io::Read};

use burn::{backend::{libtorch::LibTorchDevice, Autodiff, LibTorch}, module::Module, optim::AdamWConfig, record::CompactRecorder};
use gpt::{model::model::GPTConfig, run, tokenizer::{CharacterTokenizer, Tokenizer}, train::{self, TrainingConfig}};

type B = LibTorch<f32, i8>;
type AB = Autodiff<B>;

fn main() {

    let device = LibTorchDevice::Cpu;

    let tokenizer = CharacterTokenizer::new();

    let model= GPTConfig::new(256, tokenizer.vocab_size(), 4, 8, 512, 1024);

    let recorder = CompactRecorder::new();

    let model: gpt::model::model::GPT<B> = model.init(&device).load_file("sm-tmp/model-0", &recorder, &device)
        .expect("Error loading model");
    let optim = AdamWConfig::new();

    // let train_config = TrainingConfig::new(100, 16, 1e-3, 20, 69, model, optim);

    // let mut data_size = 1; //mb

    let (train_data, test_data) = {
        let mut f = File::open("data.txt")
            .expect("Error opening text file");
            // .take(1 * 2000); //Limit the amount of bytes you read

        let mut txt_buf = String::new();

        let _ = f.read_to_string(&mut txt_buf);

        let mut txt_train = tokenizer.encode(&txt_buf);
        let txt_test = txt_train.split_off((txt_train.len() as f32 * 0.9) as usize);

        // println!("{}", tokenizer.decode(&txt_train));

        // let train_t: Tensor<AB, 1, burn::prelude::Int> = Tensor::from_data(
        //     TensorData::new(
        //         txt_train.iter().map(|&i| i as i32).collect(), [txt_train.len()]), &device);

        // let test_t: Tensor<B, 1, burn::prelude::Int> = Tensor::from_data(
        //     TensorData::new(
        //         txt_test.iter().map(|&i| i as i32).collect(), [txt_test.len()]), &device);
        (txt_train, txt_test)
    };

    // println!("{}", tokenizer.decode(&tokenizer.format(&vec![99], 0)));
    
    // train::train::<AB>(&train_config, train_data, test_data, true, &tokenizer, device);
    
    run::run(&model, &tokenizer, 200, 128, 42);

}
