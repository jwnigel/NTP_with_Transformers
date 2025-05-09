import argparse
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import os

from dataset import get_dataloaders, get_vocab_size
from baseline_model import BaselineNTPModel, evaluate as baseline_evaluate
from transformer_model import TransformerNTPModel, evaluate as transformer_evaluate


def load_model(model_type, model_path, vocab_size, **kwargs):
    """Load a trained model."""
    if model_type == 'baseline':
        embedding_dim = kwargs.get('embedding_dim', 128)
        hidden_dims = kwargs.get('hidden_dims', [256, 128])
        
        model = BaselineNTPModel(
            vocab_size=vocab_size, 
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims
        )
        evaluate_fn = baseline_evaluate
    elif model_type == 'transformer':
        embedding_dim = kwargs.get('embedding_dim', 128)
        nhead = kwargs.get('nhead', 4)
        num_layers = kwargs.get('num_layers', 2)
        dim_feedforward = kwargs.get('dim_feedforward', 512)
        dropout = kwargs.get('dropout', 0.1)
        
        model = TransformerNTPModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        evaluate_fn = transformer_evaluate
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path))
    return model, evaluate_fn


def evaluate_model(args):
    """Evaluate a trained model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size)
    
    # Get vocabulary size
    vocab_size = get_vocab_size()
    
    # Load model
    model_kwargs = {
        'embedding_dim': args.embedding_dim,
        'hidden_dims': args.hidden_dims,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout
    }
    
    model, evaluate_fn = load_model(
        args.model, 
        args.model_path, 
        vocab_size,
        **model_kwargs
    )
    model = model.to(device)
    
    # Add timing setup
    start_time = time.time()
    
    # Evaluate with comprehensive metrics
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_fn(model, test_loader, criterion, device)
    
    # Calculate model size
    num_params = sum(p.numel() for p in model.parameters())
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(metrics['loss'])).item()
    
    # Print comprehensive results
    print(f"\nComprehensive Test Results for {args.model} model:")
    print(f"Model Size: {num_params:,} parameters")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print("\nAccuracy:")
    print(f"  Top-1: {metrics['top_k_accuracy'][1]*100:.2f}%")
    print(f"  Top-3: {metrics['top_k_accuracy'][3]*100:.2f}%")
    print(f"  Top-5: {metrics['top_k_accuracy'][5]*100:.2f}%")
    print(f"\nAverage Inference Time: {metrics['avg_inference_time']*1000:.2f}ms per batch")
    
    # Save metrics if requested
    if args.save_metrics:
        results = {
            'model_type': args.model,
            'model_size': num_params,
            'perplexity': perplexity,
            'loss': metrics['loss'],
            'accuracy': {
                'top_1': metrics['top_k_accuracy'][1],
                'top_3': metrics['top_k_accuracy'][3],
                'top_5': metrics['top_k_accuracy'][5]
            },
            'avg_inference_time_ms': metrics['avg_inference_time'] * 1000,
            'embedding_dim': args.embedding_dim,
            'batch_size': args.batch_size
        }
        
        # Add model-specific parameters
        if args.model == 'transformer':
            results.update({
                'nhead': args.nhead,
                'num_layers': args.num_layers,
                'dim_feedforward': args.dim_feedforward,
                'dropout': args.dropout
            })
        
        # Save to file
        metrics_path = f"models/{args.model}_metrics.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")
    
    # Generate predictions if requested
    if args.generate:
        generate_predictions(model, test_loader, device, num_samples=5)
    
    return metrics


def generate_predictions(model, dataloader, device, num_samples=5):
    """Generate and print some predictions."""
    print("\nSample Predictions:")
    
    # Get data samples
    batch = next(iter(dataloader))
    inputs = batch['input'][:num_samples].to(device)
    targets = batch['target'][:num_samples].to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        _, predicted = torch.max(logits, dim=1)
    
    # Print predictions
    for i in range(num_samples):
        print(f"Sample {i+1}:")
        print(f"  Input: {inputs[i].cpu().numpy()}")
        print(f"  True next token: {targets[i].item()}")
        print(f"  Predicted next token: {predicted[i].item()}")
        print(f"  Probability: {probs[i][predicted[i]].item():.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Evaluate NTP models')
    
    # Model type
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'transformer'],
                        help='Model type: baseline or transformer')
    parser.add_argument('--model_path', type=str, default='models/baseline_best.pt',
                        help='Path to the trained model')
                        
    # Common parameters
    parser.add_argument('--embedding_dim', type=int, default=128, 
                        help='Dimension of word embeddings')
    
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
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--generate', action='store_true',
                        help='Generate and print some predictions')
    parser.add_argument('--save_metrics', action='store_true',
                        help='Save evaluation metrics to file')
    
    args = parser.parse_args()
    
    evaluate_model(args)


if __name__ == "__main__":
    main() 