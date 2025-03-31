import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_dataloaders, get_vocab_size
from baseline_model import BaselineNTPModel, train_epoch as baseline_train_epoch, evaluate as baseline_evaluate
from transformer_model import TransformerNTPModel, train_epoch as transformer_train_epoch, evaluate as transformer_evaluate


def train(args):
    """Train the model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    
    # Get vocabulary size
    vocab_size = get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize model
    if args.model == 'baseline':
        model = BaselineNTPModel(
            vocab_size=vocab_size, 
            embedding_dim=args.embedding_dim,
            hidden_dims=args.hidden_dims
        )
        train_epoch_fn = baseline_train_epoch
        evaluate_fn = baseline_evaluate
    elif args.model == 'transformer':
        model = TransformerNTPModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        )
        train_epoch_fn = transformer_train_epoch
        evaluate_fn = transformer_evaluate
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize optimizer and loss function
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.0001)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_path = f"models/{args.model}_best.pt"
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch_fn(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validate
        val_loss, val_acc = evaluate_fn(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model to {best_model_path}")
    
    # Test best model
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = evaluate_fn(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }
    
    history_path = f"models/{args.model}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    # Plot training history
    plot_path = f"models/{args.model}_training.png"
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, plot_path)
    
    return model


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train NTP models')
    
    # Model type
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'transformer'],
                        help='Model type: baseline or transformer')
    
    # Common parameters
    parser.add_argument('--embedding_dim', type=int, choices=[32, 64, 128], default=128,
                      help='Embedding dimension (can be reduced for resource constraints)')
    
    # Baseline model parameters
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='Dimensions of hidden layers for baseline model')
    
    # Transformer model parameters
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads for transformer')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                        help='Dimension of feedforward network in transformer')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for transformer')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer: adam or sgd')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main() 