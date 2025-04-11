# 🔍 Attention in Transformers: Concepts and Code in PyTorch

![image](https://github.com/user-attachments/assets/de259c0c-2da3-4a1a-a78e-fa9c2f0108cc)



This project is a deep dive into the **Attention Mechanism** that powers modern Transformer architectures. It walks through the theory and implementation of self-attention, masked attention, encoder-decoder attention, and multi-headed attention — all built from scratch using PyTorch.

## 📌 Project Highlights

- ✅ **Custom implementation** of attention components in PyTorch
- 🔁 **Self-Attention**: Foundation for understanding how tokens attend to each other
- 🚫 **Masked Self-Attention**: Essential for autoregressive decoding in language models
- 🔗 **Encoder-Decoder Attention**: Facilitates input-output token alignment in sequence-to-sequence tasks
- 🧠 **Multi-Headed Attention**: Parallel attention heads for richer representation learning

---

## 🔧 Core Implementations

### 1. Self-Attention
Implements the scaled dot-product attention:
```python
scores = Q @ K.T / sqrt(d_k)
weights = softmax(scores)
output = weights @ V
```

### 2. Masked Self-Attention
Applies causal mask to prevent attending to future tokens during decoding.

### 3. Encoder-Decoder Attention
Uses encoder’s key-value with decoder’s query to align relevant input context.

### 4. Multi-Head Attention
Projects input into multiple attention heads and aggregates the outputs:

---

## 🧠 Learning Objectives

- Understand the inner workings of attention mechanisms
- Reinforce theory through hands-on PyTorch code
- Prepare for building complete transformer architectures
- Build a foundation for advanced models like BERT, GPT, and T5

---

## 📘 References

- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- PyTorch Documentation


## 🚀 Next Steps

- 🔄 Add positional encoding
- 🏗 Build a full encoder-decoder Transformer
- 🔍 Apply to NLP tasks like translation or summarization
