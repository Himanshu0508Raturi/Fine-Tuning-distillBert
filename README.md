# Fine-Tuning DistilBERT for Sentiment Analysis

> **Production-Ready Transformer Fine-Tuning with 83.6% Accuracy on SST-2 Dataset**

![License](https://img.shields.io/badge/License-MIT-green) 
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Transformers](https://img.shields.io/badge/Transformers-4.30+-orange)

## 🎯 Overview

This repository demonstrates best-in-class fine-tuning practices for transformer models using DistilBERT on the Stanford Sentiment Treebank (SST-2) dataset. The implementation achieves **83.6% validation accuracy** in under 2 minutes on a single T4 GPU while maintaining computational efficiency through parameter-efficient fine-tuning.

### Key Features

✅ **Parameter Efficiency**: Only 0.9% of model parameters are trainable (592K / 67M)  
✅ **Fast Training**: 3 epochs in ~98 seconds on T4 GPU  
✅ **Memory Optimized**: Dynamic padding and mixed-precision training  
✅ **Production Ready**: Model saving, inference pipelines, monitoring  
✅ **Best Practices**: Hyperparameter centralization, reproducibility, scalability  

---

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| Validation Accuracy | 83.60% |
| Validation F1-Score | 83.69% |
| Training Loss | 0.4017 |
| Training Time | ~98 seconds |
| Model Size | 268 MB |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Himanshu0508Raturi/Fine-Tuning-distillBert
cd Fine-Tuning-distillBert

pip install torch transformers datasets evaluate accelerate numpy
```

### Running Training

```bash
# Method 1: Python script
python train_sentiment_classifier.py

# Method 2: Jupyter notebook
jupyter notebook DistilBERT_SentimentAnalysis.ipynb
```

---

## 💻 Usage Examples

### Inference with Pipeline

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./distilbert-sst2",
    device=0
)

result = classifier("This movie is absolutely fantastic!")
print(result)
# Output: [{'label': 'LABEL_1', 'score': 0.9868}]
```

### Batch Processing

```python
texts = [
    "I loved this film!",
    "Complete waste of time.",
    "It was okay."
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{text:30s} → {result['label']} ({result['score']:.4f})")
```

---

## 🏗️ Architecture

- **Model**: DistilBERT (66.9M parameters)
- **Layers**: 6 transformer blocks (vs 12 in BERT)
- **Trainable**: Pre-classifier + Classification head only (0.9%)
- **Optimization**: Mixed-precision FP16 training, dynamic padding

---

## 📈 Training Configuration

```python
Learning Rate: 5e-5
Batch Size: 64
Epochs: 3
Max Sequence Length: 128
Warmup Ratio: 0.06
Weight Decay: 0.01
Evaluation Strategy: Per epoch
Mixed Precision: Enabled (FP16)
```

---

## 📚 Dataset

**Stanford Sentiment Treebank v2 (SST-2)**
- Task: Binary sentiment classification
- Training: 67,349 examples
- Validation: 872 examples
- Test: 1,821 examples

---

## 🔧 Hyperparameter Tuning

| Parameter | Default | For Limited VRAM | For Better Accuracy |
|-----------|---------|------------------|---------------------|
| batch_size | 64 | 16 | 128 |
| learning_rate | 5e-5 | 5e-5 | 3e-5 |
| num_epochs | 3 | 3 | 5 |
| freeze_backbone | True | True | False |
| max_length | 128 | 64 | 256 |

---

## 🖥️ Hardware Requirements

- **Minimum**: 8GB RAM (CPU, ~5-10 min/epoch)
- **Recommended**: NVIDIA T4 GPU (15GB VRAM, ~30s/epoch)
- **Optimal**: RTX 3090 or better

---

## 📖 Project Structure

```
Fine-Tuning-distillBert/
├── DistilBERT_SentimentAnalysis.ipynb    # Jupyter notebook
├── train_sentiment_classifier.py         # Standalone script
├── README.md                              # Documentation
└── distilbert-sst2/                      # Saved model (after training)
    ├── model.safetensors
    ├── tokenizer.json
    ├── config.json
    └── training_args.bin
```

---

## 🔬 Advanced Topics

### Model Quantization

```python
import torch
model = AutoModelForSequenceClassification.from_pretrained("./distilbert-sst2")
quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
# Reduces model size by 75%
```

### Multi-GPU Training

```bash
accelerate config  # Configure setup
accelerate launch train_sentiment_classifier.py
```

### Push to Hugging Face Hub

```python
trainer.push_to_hub("username/distilbert-sst2-finetuned")
# Load anywhere with: pipeline("text-classification", model="username/distilbert-sst2-finetuned")
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues or bugs
- Suggest improvements to the training pipeline
- Add support for other datasets or tasks
- Optimize for additional hardware



## 👤 Author

**Himanshu Raturi**  
GitHub: [@Himanshu0508Raturi](https://github.com/Himanshu0508Raturi)

---

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/himanshu-raturi/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/Himanshu0508Raturi)    
