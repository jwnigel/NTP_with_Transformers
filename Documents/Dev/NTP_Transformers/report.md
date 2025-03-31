# Report -- A description of code and analysis of results
## By Nigel Wright

# Baseline Model

## Word Embeddings
Used nn.Embedding from PyTorch to convert tokens ids to vectors.

## Sequence Combination
The baseline model simply averages all token embeddings in the sequence:
`combined = torch.mean(embeddings, dim=1)`

This creates a simple representation of the entire input sequence as a single vector. 

## Feed-Forward Network
The combined embedding vector is then passed through a series of feed-forward layers:
`self.feedforward = nn.Sequential(*layers)`

The network consists of:
1. First hidden layer: linear transformation from embedding_dim to 256 dimensions with ReLU activation
2. Second hidden layer: linear transformation from 256 to 128 dimensions with ReLU activation
3. Output layer: linear transformation from 128 to vocab_size dimensions

## Output Layer
The final linear layer produces logits for each token in the vocabulary, representing the model's prediction of the next token.

## Summary of baseline model
This approach is much simpler than the transformer model. It ignores token order (beyond simple averaging) and doesn't capture complex relationships between tokens. However, it serves as a  baseline to compare against the transformer approach.



# Transformer Model

## Word Embeddings
I use nn.Embedding from pytorch to store embeddings. They are scaled by `sqrt(embedding_dim)` to help with gradient flow.

## Positional Encodings
Since the transformer processes tokens simultaneously, it doesn't have any sense of position or order. That's why we need positional encodings. By using sine and cosine waves of varying frequencies, we can give words a sense of order. 

The frequencies are generated here: 
`div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))`

Then even column indices have sine applied and odd column indices cosine. 

Then a batch dimension is added with unsqueeze(0), so [seq_len, dim] becomes [1, seq_len, dim] so that the tensor can be broadcast to batches of sequences

## Transformer Encoder 

Then we do next token prediction using the transformer encoder and taken the last token (sequence_representation) to predict the following token.

## Output layer

The output layer is a linear layer to convert the transformers output from embedding dim to vocab size, and produce logits for each token in the vocab. 


# Tokenization and Data Processing

## Overview
The tokenization pipeline uses the GPT-2 tokenizer as a base and builds a custom vocabulary from the WikiText-2 dataset. It handles the conversion between text and token IDs and prepares sequence data for next token prediction.

## Vocabulary Building
1. Uses GPT-2 tokenizer to split text into tokens
2. Counts token frequencies across the entire training corpus
3. Filters out tokens with frequency < 5
4. Keeps the top 10,000 most frequent tokens
5. Adds special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`

## Sequence Preparation
For next token prediction training:
1. Each text is encoded as a sequence of token IDs
2. Creates sliding windows of 8 tokens (sequence_length)
3. For each window, the input is the 8 tokens and the target is the 9th token
4. This creates (input, target) pairs for training

## Data Pipeline
1. Load the WikiText-2 dataset (train/val/test splits)
2. Build vocabulary from training data
3. Create sequence data for all splits
4. Save processed data to JSON files for later use

This approach ensures consistent tokenization across all models and provides a clean interface for working with text data.


