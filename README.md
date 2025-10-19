# 🧠 MedQA Domain-Specific Chatbot

A **domain-specific medical question-answering chatbot** fine-tuned on the **MedQuAD dataset** using **Google’s T5 Transformer model**.  
This project aims to provide **accurate, reliable, and context-aware answers** to medical questions through a **friendly chat interface** built with **Gradio**.

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Project Structure](#-project-structure)
3. [Dataset](#-dataset)
4. [Model](#-model)
5. [Training Process](#-training-process)
6. [Application (Gradio UI)](#-application-gradio-ui)
7. [Installation](#-installation)
8. [Usage](#-usage)
9. [Results](#-results)
10. [Future Work](#-future-work)
11. [Acknowledgments](#-acknowledgments)

---

## 🚀 Overview

The **MedQA Chatbot** is a **domain-specific AI assistant** trained to answer **medical-related questions** with concise, medically accurate responses.  
It uses a fine-tuned version of **T5 (Text-to-Text Transfer Transformer)**, which is ideal for natural language understanding and question answering.

The project integrates:
- **Deep Learning (T5 fine-tuning)**
- **Natural Language Processing (NLP)**
- **Interactive User Interface (Gradio)**
- **Model Optimization (EarlyStopping + Checkpointing)**

This chatbot can be easily extended or deployed in **hospitals**, **health apps**, or **educational platforms**.

---

## 🧱 Project Structure

domain-chatbot-medqa/
│
├── data/
│ └── medquad.csv # Cleaned and formatted medical Q&A dataset
│
├── models/
│ └── t5_medqa_finetuned/ # Fine-tuned T5 model and tokenizer
│
├── notebooks/
│ └── 01_medqa_finetune_t5.ipynb # Main training and evaluation notebook
│
├── src/
│ └── app_gradio.py # Interactive chatbot UI (Gradio)
│
├── requirements.txt # All dependencies
└── README.md


---

## 🧬 Dataset

The chatbot is trained on a **custom cleaned version of the MedQuAD dataset** —  
a publicly available dataset that contains **47,000+ medical question–answer pairs** across multiple diseases, conditions, and treatments.

Each record has:
| Field | Description |
|-------|--------------|
| `question` | A natural medical question (e.g., “What causes high blood pressure?”) |
| `answer` | Verified answer from trusted medical sources (e.g., NIH, MedlinePlus) |

Before training, the data was:
- Cleaned to remove duplicates and HTML tags  
- Reformatted into `question: <text>` → `answer: <text>` pairs  
- Tokenized using `T5TokenizerFast`

---

## 🤖 Model

**Model Architecture:** T5 (Text-to-Text Transfer Transformer)  
**Base Model:** `t5-small`  
**Fine-Tuned Model Path:** `models/t5_medqa_finetuned/`  
**Framework:** TensorFlow + Hugging Face Transformers

The model is trained to map:

Input: "question: What causes high blood pressure?"
Output: "High blood pressure is caused by ..."


---

## 🧩 Training Process

Training steps included:
1. Data preprocessing and cleaning  
2. Tokenization using Hugging Face `T5TokenizerFast`  
3. Conversion to TensorFlow datasets  
4. Fine-tuning using:
   - **Adam optimizer**
   - **EarlyStopping** (monitoring validation loss)
   - **ModelCheckpoint** (saving the best model)

Example training configuration:
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss)

history = model.fit(
    train_dataset_tf,
    validation_data=test_dataset_tf,
    epochs=5,
    callbacks=[early_stopping, model_checkpoint]
)

Application (Gradio UI)

Once trained, the model was integrated into an interactive Gradio chat app located at:

src/app_gradio.py


Installation

git clone https://github.com/<your-username>/domain-chatbot-medqa.git
cd domain-chatbot-medqa

Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate    # (Mac/Linux)
venv\Scripts\activate       # (Windows)

Install dependencies
pip install -r requirements.txt
