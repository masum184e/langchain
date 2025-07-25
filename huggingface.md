# Hugging Face
Hugging Face is a company and open-source community that provides tools, models, and infrastructure for building machine learning applications, especially focused on Natural Language Processing (NLP), Computer Vision, and Audio Processing.

Hugging Face is best known for its `Transformers` library, which provides:
- Pre-trained models (like BERT, GPT, T5, etc.)
- Easy-to-use APIs for using these models in real-world applications
- Datasets and tokenizers
- Support for multiple deep learning frameworks (PyTorch, TensorFlow, JAX)
They also offer:
- **Hugging Face Hub:** A platform to share and discover models, datasets, and demos
- **Spaces:** A no-code/low-code place to deploy ML apps using Gradio or Streamlit
- **Inference API:** Scalable model hosting
- **AutoTrain:** GUI-based model training and fine-tuning

## Key Libraries in Hugging Face Ecosystem
| Library        | Purpose                                                |
| -------------- | ------------------------------------------------------ |
| `transformers` | Pretrained models for NLP, vision, audio               |
| `datasets`     | Ready-to-use datasets with preprocessing               |
| `tokenizers`   | Fast and flexible tokenizers                           |
| `evaluate`     | Common ML metrics for evaluation                       |
| `accelerate`   | Simplifies multi-GPU, mixed precision training         |
| `diffusers`    | For diffusion models like Stable Diffusion (image gen) |
| `gradio`       | Web UIs for ML models (used in Spaces)                 |

## Example
### Text Classification
```py
from transformers import pipeline

# Load a sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

# Test input
text = "Hugging Face makes working with NLP so easy!"

# Run inference
result = classifier(text)

print(result)
```

# Transormer
The Transformer is a deep learning architecture introduced in the paper “Attention Is All You Need” by Vaswani et al., 2017. It revolutionized the field of Natural Language Processing (NLP) by enabling models to process entire sequences of data in parallel rather than sequentially (like RNNs/LSTMs), using a mechanism called self-attention.

At its core, the Transformer model is based on:

- Self-Attention Mechanism
- Positional Encoding
- Layer Normalization
- Feed-Forward Networks
- Encoder-Decoder Architecture

## High Level Architecture
```css
Input → [Encoder Layer x N] → Encoder Output → [Decoder Layer x N] → Output
```
- **Encoder:** Converts input sequence into context-rich representations.
- **Decoder:** Generates the output sequence, using encoder output + previously generated tokens.

Each encoder/decoder layer includes:

- Multi-Head Self-Attention
- Feed-Forward Neural Network
- Add & Norm layers

## Example
### Sentiment Analysis
```py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Input text
text = "I love Hugging Face! It makes NLP so much easier."

# Tokenize input
inputs = tokenizer(text, return_tensors="pt")

# Get model prediction
with torch.no_grad():
    outputs = model(**inputs)

# Convert logits to probabilities
probs = torch.nn.functional.softmax(outputs.logits, dim=1)

# Output prediction
print(f"Sentiment probabilities: {probs}")
```
- `AutoTokenizer`: Turns text into numerical input (token IDs, attention masks).
- `AutoModelForSequenceClassification`: A pretrained Transformer (like DistilBERT) adapted for classification.
- `outputs.logits`: Raw scores for each class (e.g., positive/negative sentiment).
- `softmax`: Converts logits into interpretable probabilities.
## Why Transformers Are Powerful
| Feature                 | Benefit                                                       |
| ----------------------- | ------------------------------------------------------------- |
| Parallel Processing     | Unlike RNNs, Transformers process sequences in parallel.      |
| Long-range Dependencies | Self-attention captures relationships between distant tokens. |
| Transfer Learning       | Pretrained models can be fine-tuned on various tasks.         |
| Scalability             | Basis for large models (GPT, BERT, T5, etc.).                 |

# Pipeline
