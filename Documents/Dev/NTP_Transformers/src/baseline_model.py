import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class BaselineNTPModel(nn.Module):
    """Baseline Next Token Prediction Model.
    
    This model uses word embeddings and feed-forward layers to predict the next token.
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dims=[256, 128]):
        """
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the word embeddings
            hidden_dims: List of hidden layer dimensions
        """
        super(BaselineNTPModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Define the feed-forward layers
        layers = []
        
        # Input layer - processes the combined embedding
        input_dim = embedding_dim  # If using mean/max pooling
        # input_dim = embedding_dim * seq_length  # If using concatenation
        
        # Add hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, vocab_size))
        
        self.feedforward = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input tokens [batch_size, seq_length]
        
        Returns:
            logits: Prediction logits [batch_size, vocab_size]
        """
        # Get embeddings
        embeddings = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        
        # Combine embeddings by averaging
        combined = torch.mean(embeddings, dim=1)  # [batch_size, embedding_dim]
        
        # Alternative: Combine by max pooling
        # combined = torch.max(embeddings, dim=1)[0]  # [batch_size, embedding_dim]
        
        # Alternative: Combine by concatenation
        # combined = embeddings.view(embeddings.size(0), -1)  # [batch_size, seq_length * embedding_dim]
        
        # Pass through feed-forward layers
        logits = self.feedforward(combined)  # [batch_size, vocab_size]
        
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


def evaluate(model, dataloader, criterion, device, k_values=[1, 3, 5]):
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0
    total = 0
    correct_topk = {k: 0 for k in k_values}
    inference_times = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Measure inference time
            start_time = time.time()
            logits = model(inputs)
            inference_times.append(time.time() - start_time)
            
            # Calculate loss
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            # Calculate top-k accuracy
            for k in k_values:
                _, topk_indices = torch.topk(logits, k, dim=1)
                correct_topk[k] += torch.any(topk_indices == targets.unsqueeze(1), dim=1).sum().item()
            
            total += targets.size(0)
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    topk_accuracy = {k: correct_topk[k] / total for k in k_values}
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'top_k_accuracy': topk_accuracy,
        'avg_inference_time': avg_inference_time
    } 