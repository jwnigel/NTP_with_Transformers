import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        """
        Args:
            d_model: Embedding dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional embedding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even idx
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd idx
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input embeddings [batch_size, seq_length, embedding_dim]
        
        Returns:
            Embeddings with positional encoding [batch_size, seq_length, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerNTPModel(nn.Module):
    """Transformer-based Next Token Prediction Model."""
    
    def __init__(self, vocab_size, embedding_dim=128, nhead=4, num_layers=2, 
                 dim_feedforward=512, dropout=0.1, max_seq_length=5000):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feed-forward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super(TransformerNTPModel, self).__init__()
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            embedding_dim, max_seq_length, dropout
        )
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x):
        """
        Args:
            x: Input tokens [batch_size, seq_length]
        
        Returns:
            logits: Prediction logits [batch_size, vocab_size]
        """
        # Create mask for transformer (all tokens can attend to all other tokens)
        src_mask = None
        
        # Embed tokens
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)  # [batch_size, seq_length, embedding_dim]
        
        # Add positional encoding
        x = self.positional_encoding(x)  # [batch_size, seq_length, embedding_dim]
        
        # Pass through transformer encoder
        # We want to predict the next token after the sequence, so we use the representation
        # of the last token in the sequence
        transformer_output = self.transformer_encoder(x, src_mask)  # [batch_size, seq_length, embedding_dim]
        sequence_representation = transformer_output[:, -1, :]  # [batch_size, embedding_dim]
        
        # Pass through output layer
        logits = self.output_layer(sequence_representation)  # [batch_size, vocab_size]
        
        return logits


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        logits = model(inputs)
        loss = criterion(logits, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track statistics
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, targets)
            
            # Track statistics
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    return total_loss / len(dataloader), correct / total 