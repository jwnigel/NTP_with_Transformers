import os
import json
from collections import Counter
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

class NTPTokenizer:
    def __init__(self, vocab_size=10000, min_freq=5, seq_length=8):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.seq_length = seq_length
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Special tokens
        self.special_tokens = {
            "PAD": "<PAD>",
            "UNK": "<UNK>",
            "BOS": "<BOS>",
            "EOS": "<EOS>"
        }
        
        # Initialize dictionaries
        self.token2idx = {}
        self.idx2token = {}
        self.vocab_built = False
        
    def build_vocab(self, texts):
        """Build vocabulary from texts."""
        print("Building vocabulary...")
        # Count token frequencies
        counter = Counter()
        
        # Tokenize texts and count tokens
        for text in tqdm(texts):
            tokens = self.tokenizer.tokenize(text)
            counter.update(tokens)
        
        # Filter tokens with frequency < min_freq
        filtered_tokens = [(token, count) for token, count in counter.items() 
                          if count >= self.min_freq]
        
        # Sort by frequency and take top vocab_size - num_special_tokens
        sorted_tokens = sorted(filtered_tokens, key=lambda x: x[1], reverse=True)
        sorted_tokens = sorted_tokens[:self.vocab_size - len(self.special_tokens)]
        
        # Create token to index mappings
        self.token2idx = {}
        
        # Add special tokens first
        for i, token in enumerate(self.special_tokens.values()):
            self.token2idx[token] = i
        
        # Add remaining tokens
        for i, (token, _) in enumerate(sorted_tokens):
            self.token2idx[token] = i + len(self.special_tokens)
            
        # Create reverse mapping
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        
        self.vocab_built = True
        print(f"Vocabulary built with {len(self.token2idx)} tokens.")
        
    def save_vocab(self, path):
        """Save vocabulary to a file."""
        vocab_data = {
            "token2idx": self.token2idx,
            "idx2token": {int(k): v for k, v in self.idx2token.items()},
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "seq_length": self.seq_length
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"Vocabulary saved to {path}")
        
    def load_vocab(self, path):
        """Load vocabulary from a file."""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
            
        self.token2idx = vocab_data["token2idx"]
        self.idx2token = {int(k): v for k, v in vocab_data["idx2token"].items()}
        self.special_tokens = vocab_data["special_tokens"]
        self.vocab_size = vocab_data["vocab_size"]
        self.min_freq = vocab_data["min_freq"]
        self.seq_length = vocab_data["seq_length"]
        self.vocab_built = True
        
        print(f"Vocabulary loaded from {path} with {len(self.token2idx)} tokens.")
        
    def encode(self, text):
        """Convert text to token indices."""
        if not self.vocab_built:
            raise ValueError("Vocabulary has not been built yet. Call build_vocab first.")
            
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        
        # Convert tokens to indices
        indices = []
        for token in tokens:
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                indices.append(self.token2idx[self.special_tokens["UNK"]])
                
        return indices
    
    def decode(self, indices):
        """Convert token indices to text."""
        if not self.vocab_built:
            raise ValueError("Vocabulary has not been built yet. Call build_vocab first.")
            
        tokens = [self.idx2token.get(idx, self.special_tokens["UNK"]) for idx in indices]
        return self.tokenizer.convert_tokens_to_string(tokens)
    
    def prepare_sequence_data(self, texts):
        """Prepare sequence data for next token prediction."""
        if not self.vocab_built:
            raise ValueError("Vocabulary has not been built yet. Call build_vocab first.")
            
        inputs = []
        targets = []
        
        for text in tqdm(texts, desc="Preparing sequences"):
            # Encode full text
            token_ids = self.encode(text)
            
            # Create sequences of length seq_length with next token as target
            for i in range(len(token_ids) - self.seq_length):
                seq = token_ids[i:i + self.seq_length]
                next_token = token_ids[i + self.seq_length]
                
                inputs.append(seq)
                targets.append(next_token)
                
        return inputs, targets


def main():
    """Download dataset and build vocabulary."""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Initialize tokenizer
    tokenizer = NTPTokenizer(vocab_size=10000, min_freq=5, seq_length=8)
    
    # Build vocabulary from training data
    tokenizer.build_vocab(dataset["train"]["text"])
    
    # Save vocabulary
    tokenizer.save_vocab("data/vocab.json")
    
    # Prepare sequence data
    print("Preparing training data...")
    train_inputs, train_targets = tokenizer.prepare_sequence_data(dataset["train"]["text"])
    
    print("Preparing validation data...")
    val_inputs, val_targets = tokenizer.prepare_sequence_data(dataset["validation"]["text"])
    
    print("Preparing test data...")
    test_inputs, test_targets = tokenizer.prepare_sequence_data(dataset["test"]["text"])
    
    # Save processed data
    print("Saving processed data...")
    os.makedirs("data", exist_ok=True)
    
    # Save data as JSON for simplicity
    with open("data/train_data.json", "w") as f:
        json.dump({"inputs": train_inputs, "targets": train_targets}, f)
        
    with open("data/val_data.json", "w") as f:
        json.dump({"inputs": val_inputs, "targets": val_targets}, f)
        
    with open("data/test_data.json", "w") as f:
        json.dump({"inputs": test_inputs, "targets": test_targets}, f)
    
    print("Data preparation completed!")


if __name__ == "__main__":
    main() 