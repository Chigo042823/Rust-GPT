use std::collections::HashMap;

use burn::{module::Module, prelude::Backend, tensor::{Tensor, TensorData}};
use unicode_segmentation::UnicodeSegmentation;

const START_TOKEN: &str = "<|SOS|>";
const END_TOKEN: &str = "<|EOS|>";
const UNK_TOKEN: &str = "<|?|>";
const PADDING_TOKEN: &str = "<|PAD|>";

pub trait Tokenizer {
    fn encode(&self, input: &str) -> Vec<usize>;
    fn format(&self, input: &[usize], max_seq_len: usize) -> Vec<usize>;
    fn decode(&self, input: &Vec<usize>) -> String;
    fn vocab_size(&self) -> usize;
}

pub struct CharacterTokenizer {
    pub cti_vocab: HashMap<String, usize>,
    pub itc_vocab: HashMap<usize, String>
}

impl CharacterTokenizer {
    pub fn new() -> Self {
        let mut vocab_str: Vec<String> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'()[]{}-_+=*/|@#$%^&`~<>"
            .chars()
            .map(|c| c.to_string())
            .collect();

        vocab_str.push("\"".to_owned());
        vocab_str.push("\\".to_owned());
        vocab_str.push("\n".to_owned());
        vocab_str.push("\r".to_owned());
        vocab_str.push("\t".to_owned());
        
        let vocab_size = vocab_str.len();

        let mut itc_vocab: HashMap<usize, String> = vocab_str
            .iter()
            .enumerate()
            .map(|(k, v)| (k + 1, v.clone()))
            .collect();

        itc_vocab.insert(0, START_TOKEN.to_string());
        itc_vocab.insert(vocab_size + 1, PADDING_TOKEN.to_string());
        itc_vocab.insert(vocab_size + 2, UNK_TOKEN.to_string());
        itc_vocab.insert(vocab_size + 3, END_TOKEN.to_string());
        
        let mut cti_vocab: HashMap<String, usize> = vocab_str
            .iter()
            .enumerate()
            .map(|(k, v)| (v.clone(), k + 1))
            .collect();

        cti_vocab.insert(START_TOKEN.to_string(), 0);
        cti_vocab.insert( PADDING_TOKEN.to_string(), vocab_size + 1);
        cti_vocab.insert( UNK_TOKEN.to_string(), vocab_size + 2);
        cti_vocab.insert(END_TOKEN.to_string(), vocab_size + 3);

        Self {
            itc_vocab,
            cti_vocab
        }
    }
    pub fn print_vocab(&self) {
        println!("{:#?}", self.cti_vocab);
        println!("Vocab Size: {}", self.vocab_size());
    }
}

impl Tokenizer for CharacterTokenizer {
    fn encode(&self, input: &str) -> Vec<usize> {
       let enc: Vec<usize> = input
        .chars()
        .map(|c| {
            self.cti_vocab
                .get(&c.to_string())
                .copied()
                .unwrap_or_else(|| self.cti_vocab[UNK_TOKEN])
        })
        .collect();
        enc
    }

    fn decode(&self, input: &Vec<usize>) -> String {
        input
            .iter()
            .map(|i| 
                self.itc_vocab.get(&i).expect(&format!("Could not find {} in vocab dictionary", i)).clone()
            )
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.cti_vocab.len()
    }
    
    fn format(&self, input: &[usize], max_seq_len: usize) -> Vec<usize> {
        let tmp = input.to_vec();
        let inp_len = tmp.len();
        let mut enc = tmp.clone();
        
        enc.insert(0, 0);
        enc.push(self.vocab_size() - 1);

        // for _ in 2..(inp_len.abs_diff(max_seq_len)) {
        //     enc.push(self.vocab_size() - 3);
        // }

        enc
    }
}