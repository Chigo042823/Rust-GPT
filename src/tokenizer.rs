use std::collections::HashMap;

use burn::{module::Module, prelude::Backend, tensor::{Tensor, TensorData}};

pub trait Tokenizer {
    fn encode(&self, input: &str) -> Vec<usize>;
    fn decode(&self, input: &Vec<usize>) -> String;
    fn vocab_size(&self) -> usize;
}

pub struct CharacterTokenizer {
    pub cti_vocab: HashMap<String, usize>,
    pub itc_vocab: HashMap<usize, String>
}

impl CharacterTokenizer {
    pub fn new() -> Self {
        let vocab_str: Vec<String> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()[]{}-_+=*/\\|@#$%^&`~<>\n\t"
            .chars()
            .map(|c| c.to_string())
            .collect();
        
        let vocab_size = vocab_str.len();

        let mut itc_vocab: HashMap<usize, String> = vocab_str
            .iter()
            .enumerate()
            .map(|(k, v)| (k + 1, v.clone()))
            .collect();

        itc_vocab.insert(0, "<S>".to_string());
        itc_vocab.insert(vocab_size + 1, "<P>".to_string());
        itc_vocab.insert(vocab_size + 2, "<E>".to_string());
        
        let mut cti_vocab: HashMap<String, usize> = vocab_str
            .iter()
            .enumerate()
            .map(|(k, v)| (v.clone(), k + 1))
            .collect();

        cti_vocab.insert("<S>".to_string(), 0);
        cti_vocab.insert( "<P>".to_string(), vocab_size + 1);
        cti_vocab.insert("<E>".to_string(), vocab_size + 2);

        Self {
            itc_vocab,
            cti_vocab
        }
    }
}

impl Tokenizer for CharacterTokenizer {
    fn encode(&self, input: &str) -> Vec<usize> {
        let strs = ["<S>", input, "<E>"].join("");
        strs
            .chars()
            .map(|c| 
                self.cti_vocab.get(&c.to_string()).copied().expect(&format!("Could not find {} in vocab dictionary", c))
            )
            .collect()
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
}