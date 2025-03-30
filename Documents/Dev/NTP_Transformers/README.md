# Next Token Prediction with Transformers

This project implements and compares two approaches to next-token prediction:
1. A baseline model using word embeddings and feed-forward layers
2. A multi-head Transformer-based model

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

- `data/`: Contains the WikiText-2 dataset
- `src/`: Source code
  - `tokenizer.py`: Tokenization and preprocessing
  - `dataset.py`: Dataset preparation and loading
  - `baseline_model.py`: Baseline next-token prediction model
  - `transformer_model.py`: Transformer-based next-token prediction model
  - `train.py`: Training script
  - `evaluate.py`: Evaluation script
- `notebooks/`: Jupyter notebooks for exploration and visualization

## Usage

1. Prepare the dataset:
```
python src/dataset.py
```

2. Train the baseline model:
```
python src/train.py --model baseline
```

3. Train the transformer model:
```
python src/train.py --model transformer
```

4. Evaluate models:
```
python src/evaluate.py --model baseline
python src/evaluate.py --model transformer
``` 