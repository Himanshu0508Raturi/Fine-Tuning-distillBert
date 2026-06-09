# Fine-Tuning DistilBERT for Sentiment Analysis

A production-ready implementation of fine-tuning **DistilBERT** on the **SST-2 (Stanford Sentiment Treebank)** dataset for binary sentiment classification. This repository demonstrates best practices in modern NLP with the Hugging Face transformers library.

## 📋 Overview

This project implements sentiment analysis by fine-tuning a lightweight DistilBERT model on the SST-2 dataset. Key features include:

- **Efficient Architecture**: Uses DistilBERT (66.9M parameters) instead of full BERT, reducing memory and computation by ~40%
- **Selective Fine-tuning**: Freezes the backbone transformer and trains only the classification head (~0.9% of parameters)
- **Modern Best Practices**: Dynamic padding, mixed-precision training, and structured hyperparameter management
- **High Performance**: Achieves **83.6% validation accuracy** in under 2 minutes on a single T4 GPU

## 🎯 Results

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | 83.60% |
| **Validation F1-Score** | 83.69% |
| **Training Loss** | 0.4017 |
| **Training Time** | ~98 seconds |
| **Total Steps** | 3,159 |

### Sample Predictions

```
Sentence: "The movie was absolutely brilliant and left me in tears of joy."
Prediction: POSITIVE (0.9868)

Sentence: "A complete waste of two hours. The plot made no sense whatsoever."
Prediction: NEGATIVE (0.9755)

Sentence: "Not bad, but the second half dragged on a bit."
Prediction: NEGATIVE (0.5730)
```

## 📊 Dataset

**SST-2 (Stanford Sentiment Treebank)**
- **Task**: Binary sentiment classification (positive/negative)
- **Training Set**: 67,349 examples
- **Validation Set**: 872 examples
- **Test Set**: 1,821 examples
- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets/stanfordnlp/sst2)

Sample training example:
```python
{'idx': 0, 'sentence': 'hide new secretions from the parental units ', 'label': 0}
```

## 🏗️ Architecture

### Model Configuration

```python
MODEL_NAME       = "distilbert-base-uncased"   # Base checkpoint
DATASET_NAME     = "stanfordnlp/sst2"          # HF dataset ID
NUM_LABELS       = 2                            # Positive / Negative
MAX_LENGTH       = 128                          # Tokenizer truncation
BATCH_SIZE       = 64                           # Per-device batch size
LEARNING_RATE    = 5e-5
NUM_EPOCHS       = 3
FREEZE_BACKBONE  = True                         # Train classifier head only
OUTPUT_DIR       = "./distilbert-sst2"
```

### Parameter Efficiency

```
Trainable params: 592,130 / 66,955,010 (0.9%)
```

By freezing the DistilBERT backbone and only training the pre-classifier and classification layers, we reduce:
- GPU memory footprint
- Training time
- Risk of overfitting on small datasets

## 🚀 Quick Start

### Installation

```bash
pip install -q transformers datasets evaluate accelerate torch
```

### Running the Notebook

```bash
# Download and prepare the dataset
python -c "from datasets import load_dataset; load_dataset('stanfordnlp/sst2')"

# Run training
jupyter notebook DistilBERT_SentimentAnalysis.ipynb
```

### Training Pipeline

The implementation follows this workflow:

1. **Load Model & Tokenizer**
   ```python
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   
   tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
   model = AutoModelForSequenceClassification.from_pretrained(
       "distilbert-base-uncased",
       num_labels=2
   )
   ```

2. **Prepare Data**
   ```python
   from transformers import DataCollatorWithPadding
   
   data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
   # Dynamic padding per batch: reduces wasted computation
   ```

3. **Configure Training**
   ```python
   from transformers import TrainingArguments, Trainer
   
   training_args = TrainingArguments(
       output_dir="./distilbert-sst2",
       num_train_epochs=3,
       per_device_train_batch_size=64,
       learning_rate=5e-5,
       fp16=True,  # Mixed precision training
       eval_strategy="epoch",
       save_strategy="epoch",
       load_best_model_at_end=True,
   )
   ```

4. **Train & Evaluate**
   ```python
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_data,
       eval_dataset=val_data,
       data_collator=data_collator,
       compute_metrics=compute_metrics,
   )
   trainer.train()
   ```

## 📈 Training Metrics

### Epoch-wise Performance

| Epoch | Training Loss | Validation Loss | Accuracy | F1-Score |
|-------|---------------|-----------------|----------|----------|
| 1     | 0.3770        | 0.3834          | 0.8303   | 0.8311   |
| 2     | 0.3775        | 0.3755          | **0.8360** | **0.8369** |
| 3     | 0.3611        | 0.3748          | 0.8337   | 0.8358   |

**Best model**: Epoch 2 (selected via `load_best_model_at_end=True`)

## 🔧 Key Implementation Details

### Dynamic Padding

```python
def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=128,
        # No padding here — DataCollatorWithPadding handles it dynamically
    )
```

**Benefit**: Reduces memory usage by padding only to the length of the longest sequence in each batch, not globally to `max_length=128`.

### Mixed-Precision Training

```python
fp16=torch.cuda.is_available()  # Automatically enables on GPU
```

- Reduces GPU memory by ~50%
- Speeds up training by ~2x (on supported hardware)
- Maintains accuracy through gradient scaling

### Evaluation Metrics

```python
import evaluate

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}
```

## 💾 Model Output

The fine-tuned model is saved to `./distilbert-sst2/` with the following structure:

```
distilbert-sst2/
├── config.json                    # Model configuration
├── model.safetensors             # Model weights (267.8 MB)
├── tokenizer.json                # Tokenizer vocabulary
├── tokenizer_config.json         # Tokenizer settings
├── training_args.bin             # Training hyperparameters
└── checkpoint-*                  # Intermediate checkpoints
```

## 🎭 Usage: Inference

### Using the Transformers Pipeline

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./distilbert-sst2",
    device=0  # GPU device
)

results = classifier([
    "The movie was absolutely brilliant!",
    "A complete waste of time."
])

# Output:
# [{'label': 'LABEL_1', 'score': 0.9868},
#  {'label': 'LABEL_0', 'score': 0.9755}]
```

### Direct Model Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("./distilbert-sst2")
model = AutoModelForSequenceClassification.from_pretrained("./distilbert-sst2")

text = "This movie is wonderful!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
prediction = torch.argmax(logits, dim=-1)
print(f"Sentiment: {'Positive' if prediction.item() == 1 else 'Negative'}")
```

## 🖥️ Hardware Requirements

### Minimum (CPU-only)
- 8GB RAM
- ~3 minutes training time

### Recommended (GPU)
- **GPU**: NVIDIA T4 or better (15.6 GB VRAM)
- **Training Time**: ~98 seconds
- **Mixed Precision**: fp16 reduces memory by 50%

## 📚 Dependencies

```
transformers>=4.30.0
datasets>=2.12.0
evaluate>=0.4.0
accelerate>=0.20.0
torch>=2.0.0
numpy
```

Install all at once:
```bash
pip install transformers datasets evaluate accelerate torch
```

## 🔍 Model Details

### DistilBERT Architecture
- **Layers**: 6 transformer blocks (vs 12 in BERT)
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Total Parameters**: 66.9M (40% smaller than BERT)
- **Distillation**: Trained via knowledge distillation from BERT-base

### Classification Head (Trainable)
- **Pre-classifier**: Linear layer (768 → 768)
- **Dropout**: 0.2
- **Classifier**: Linear layer (768 → 2)
- **Activation**: ReLU in pre-classifier, Softmax in classifier

## 📖 References

- **Paper**: [DistilBERT: A distilled version of BERT](https://arxiv.org/abs/1910.01108)
- **Dataset**: [SST-2 on Hugging Face](https://huggingface.co/datasets/stanfordnlp/sst2)
- **Transformers Library**: [Hugging Face Transformers](https://huggingface.co/transformers/)
- **Trainer Class Docs**: [Hugging Face Trainer Documentation](https://huggingface.co/docs/transformers/training)

## 📝 Notebook Cells Overview

| Cell | Description | Key Output |
|------|-------------|-----------|
| 1 | Install dependencies | `transformers`, `datasets`, `evaluate`, `accelerate` |
| 2 | Setup environment & detect GPU | `Using device: cuda` (Tesla T4, 15.6 GB) |
| 3 | Define hyperparameters | All config in one place for easy tuning |
| 4 | Load SST-2 dataset | 67,349 train / 872 val / 1,821 test examples |
| 5 | Tokenize dataset | Dynamic padding enabled |
| 6 | Create data collator | Batch shape: `[4, 15]` (after padding) |
| 7 | Load DistilBERT & freeze | Trainable: 592K / 67M (0.9%) |
| 8 | Define metrics | Accuracy + F1-Score |
| 9 | Configure training | Learning rate, batch size, epochs |
| 10 | Initialize Trainer | Setup with all components |
| 11 | Train model | 3 epochs in 98.4 seconds |
| 12 | Evaluate | Val accuracy: 83.60% |
| 13 | Save model | Saved to `./distilbert-sst2/` |
| 14-15 | Inference examples | Test with custom sentences |

## 🎓 Learning Outcomes

This project demonstrates:

✅ **Modern NLP Workflow**: Loading, tokenizing, and fine-tuning transformers  
✅ **Efficient Training**: Selective fine-tuning, dynamic padding, mixed precision  
✅ **Best Practices**: Hyperparameter centralization, evaluation metrics, checkpointing  
✅ **Production Ready**: Model saving/loading, inference pipelines, reproducibility  
✅ **GPU Optimization**: Automatic device detection, mixed-precision fp16  

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues or bugs
- Suggest improvements to the training pipeline
- Add support for other datasets or tasks
- Optimize for additional hardware



## 👨‍💻 Author

**Himanshu Raturi**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/himanshu-raturi/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/Himanshu0508Raturi)    