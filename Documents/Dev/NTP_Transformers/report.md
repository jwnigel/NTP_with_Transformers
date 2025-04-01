# Report -- A description of code and analysis of results
## By Nigel Wright



# Tokenization and Data Processing

The tokenization pipeline uses the GPT-2 tokenizer as a base and builds a custom vocabulary from the WikiText-2 dataset. It handles conversion between text and token IDs and prepares sequence data for next token prediction.

## Vocab Building
1. Uses GPT-2 tokenizer to split text into tokens
2. Counts token frequencies across the entire training corpus
3. Filters out tokens with frequency < 5
4. Keeps the top 10,000 most frequent tokens
5. Adds special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`

## Sequence Prep
For next token prediction training:
1. Each text is encoded as a sequence of token IDs
2. Creates sliding windows of 8 tokens (sequence_length)
3. For each window, the input is the 8 tokens and the target is the 9th token
4. This creates (input, target) pairs for training

## Pipeline
1. Load the WikiText-2 dataset (train/val/test splits)
2. Build vocabulary from training data
3. Create sequence data for all splits
4. Save processed data to JSON files for later use


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
This approach is much simpler than the transformer model. It ignores token order (beyond simple averaging) and doesn't capture complex relationships between tokens. However, it serves as a  baseline to compare against the transformer approach and can be compared to a Continuous Bag of Words model (CBOW). 



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

Then we do next token prediction using the transformer encoder and take the last token (sequence_representation) to predict the following token.

## Output layer

The output layer is a linear layer to convert the transformers output from embedding dim to vocab size, and produce logits for each token in the vocab. 


# Analysis of Results

## Model Architectures
### Baseline Model
- Simple feed-forward architecture
- Parameters: 1,280,000 (10,000 * 128 + 128 * 256 + 256 * 128 + 128 * 10,000)
- Embedding dimension: 128
- Hidden layers: [256, 128]
- Loss function: Cross-entropy
- Optimizer: Adam (lr=0.0001)

### Transformer Model
- Transformer encoder architecture
- Parameters: 2,560,000 (10,000 * 128 + 128 * 128 * 4 + 128 * 512 + 512 * 128 + 128 * 10,000)
- Embedding dimension: 128
- Attention heads: 4
- Transformer layers: 2
- Feed-forward dimension: 512
- Dropout: 0.1

## Performance Metrics
### Training Dynamics
Both models were trained with:
- Batch size: 64
- Sequence length: 8
- Vocabulary size: 10,000
- WikiText-2 dataset

### Evaluation Metrics
For each model, we measure:
1. Perplexity on test set
2. Top-k accuracy (k=1,3,5)
3. Average inference time
4. Model size

### Results
Baseline Model Results:
- Test Perplexity: 360.5 (exp(5.890))
- Test Accuracy: 15.59%
- Final Training Loss: 5.675
- Final Validation Loss: 5.974
- Parameters: 1.28M

Transformer Model Results:
- Test Perplexity: 120.5 (exp(4.792))
- Test Accuracy: 24.16%
- Final Training Loss: 4.676
- Final Validation Loss: 4.801
- Parameters: 2.56M

## Analysis

### Model Comparison
1. **Accuracy vs Complexity**
   - Transformer achieves 55% better accuracy (24.16% vs 15.59%)
   - Requires 2x more parameters (2.56M vs 1.28M)
   - Lower perplexity (120.5 vs 360.5) indicates better language modeling

2. **Training Dynamics**
   - Transformer converges faster and to better minima
   - Baseline model shows signs of overfitting (validation loss increases after epoch 4)
   - Transformer maintains stable validation loss throughout training

3. **Resource Requirements**
   - Baseline model: 1.28M parameters, simpler architecture
   - Transformer model: 2.56M parameters, more complex architecture
   - Transformer requires more memory and computation but achieves better results

### Key Findings
1. Transformer architecture significantly outperforms the baseline model in both accuracy and perplexity
2. The baseline model's simple averaging approach is insufficient for capturing complex language patterns
3. Transformer's self-attention mechanism effectively captures long-range dependencies
4. The trade-off between model complexity and performance favors the transformer approach

### Limitations
1. Limited vocabulary size (10,000 tokens)
2. Fixed sequence length (8 tokens)
3. Small model size compared to state-of-the-art transformers
4. Training on CPU only (no GPU acceleration)

## Future Improvements
1. Experiment with different sequence lengths
2. Try larger vocabulary sizes
3. Test different embedding dimensions
4. Add GPU support for faster training
5. Implement beam search for better inference
6. Add layer normalization and residual connections to baseline model

