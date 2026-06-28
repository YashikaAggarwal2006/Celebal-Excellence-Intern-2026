# Deep Learning Text Generation using Vanilla RNN, LSTM, and GRU

## Overview

This project demonstrates text generation using three recurrent neural network architectures:

* Vanilla RNN (SimpleRNN)
* Long Short-Term Memory (LSTM)
* Gated Recurrent Unit (GRU)

The models are trained on a custom text corpus to learn grammar, contextual relationships, and next-word prediction. Their performance is compared based on training loss and generated text quality.

---

## Objectives

* Learn text preprocessing and tokenization.
* Create n-gram sequences for next-word prediction.
* Train and compare Vanilla RNN, LSTM, and GRU models.
* Generate coherent text from a common seed phrase.
* Analyze training loss across different sequence models.

---

## Features

* Custom text corpus
* Text tokenization using Keras Tokenizer
* N-gram sequence generation
* Sequence padding using `pad_sequences`
* Embedding layer for word representation
* Vanilla RNN implementation
* LSTM implementation
* GRU implementation
* Training loss comparison graph
* Next-word text generation
* Comparison of sequence learning models

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib

---

## Project Workflow

1. Load custom text corpus.
2. Tokenize text into integer sequences.
3. Generate progressive n-gram sequences.
4. Apply sequence padding.
5. Split data into input (`X`) and output (`y`).
6. Train Vanilla RNN model.
7. Train LSTM model.
8. Train GRU model.
9. Compare training loss.
10. Generate text from a common seed phrase.

---

## Student Customization Tasks Completed

* Replaced the default corpus with a custom paragraph.
* Increased embedding dimension from **32** to **64**.
* Increased hidden units from **64** to **128**.
* Increased training epochs from **100** to **200**.
* Generated **10** words instead of **5**.

---



## Results

The project compares the three models using:

* Cross-entropy training loss
* Learning convergence
* Generated text quality
* Ability to capture contextual dependencies

Generally:

* Vanilla RNN learns quickly but struggles with long-term dependencies.
* LSTM produces more coherent text by preserving long-term information.
* GRU provides performance comparable to LSTM with fewer parameters and faster training.


**Yashika Aggarwal**
