import os
import json
import torch
from torch.utils.data import Dataset, DataLoader


class NTPDataset(Dataset):
    """Next Token Prediction Dataset"""
    
    def __init__(self, inputs, targets, pad_idx=0):
        """
        Args:
            inputs: List of input sequences
            targets: List of target tokens
            pad_idx: Index of padding token
        """
        self.inputs = inputs
        self.targets = targets
        self.pad_idx = pad_idx
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'input': torch.tensor(self.inputs[idx], dtype=torch.long),
            'target': torch.tensor(self.targets[idx], dtype=torch.long)
        }


def load_data(data_path):
    """Load data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data['inputs'], data['targets']


def get_dataloaders(batch_size=64, data_dir='data'):
    """Create dataloaders for training, validation and testing."""
    # Load data
    train_inputs, train_targets = load_data(os.path.join(data_dir, 'train_data.json'))
    val_inputs, val_targets = load_data(os.path.join(data_dir, 'val_data.json'))
    test_inputs, test_targets = load_data(os.path.join(data_dir, 'test_data.json'))
    
    # Load vocabulary to get PAD token index
    with open(os.path.join(data_dir, 'vocab.json'), 'r') as f:
        vocab_data = json.load(f)
    pad_idx = vocab_data['token2idx']['<PAD>']
    
    # Create datasets
    train_dataset = NTPDataset(train_inputs, train_targets, pad_idx)
    val_dataset = NTPDataset(val_inputs, val_targets, pad_idx)
    test_dataset = NTPDataset(test_inputs, test_targets, pad_idx)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size
    )
    
    return train_loader, val_loader, test_loader


def get_vocab_size(data_dir='data'):
    """Get vocabulary size from vocab file."""
    with open(os.path.join(data_dir, 'vocab.json'), 'r') as f:
        vocab_data = json.load(f)
    return len(vocab_data['token2idx'])


def main():
    """Test data loading."""
    if not os.path.exists('data/train_data.json'):
        print("Data files not found. Please run src/tokenizer.py first.")
        return
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"Input shape: {batch['input'].shape}")
    print(f"Target shape: {batch['target'].shape}")
    
    # Get vocabulary size
    vocab_size = get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")


if __name__ == "__main__":
    main() 