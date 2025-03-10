# Multi-Task Learning with Transformers 
A PyTorch implementation of a multi-task learning model that performs **binary sentence classification** and **sentiment analysis** using a pre-trained transformer as the backbone.

---

## **Table of Contents**
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Inference](#inference)
- [Results](#results)
- [Future Work](#future-work)

---

## **Introduction**
This project demonstrates how to:
- Use **transformers** for sentence embeddings.
- Implement **multi-task learning** with task-specific heads:
   - **Task A:** Binary classification (positive vs. negative).
   - **Task B:** Sentiment analysis (positive, neutral, negative).
- Fine-tune task-specific heads while **freezing the transformer layers**.

---

## **Model Architecture**
- **Base Model:** `SentenceTransformer` (`all-MiniLM-L6-v2`).
- **Task-Specific Heads:**
   - `classifier_head`: Classifies sentences as **positive** or **negative**.
   - `sentiment_head`: Classifies sentences into **positive, neutral, or negative**.
- **Loss Functions:**  
   - `CrossEntropyLoss` for both tasks.
- **Optimizer:**  
   - `Adam` with a learning rate of `0.001`.

---

## **Dependencies**
- `torch` (PyTorch)
- `sentence-transformers`

---

## **Installation**
```bash
pip install -r requirements.txt
