# Sentiment Analysis of Peer Reviews

This project involves building a machine learning model to predict the sentiment of peer reviews from the ICLR 2017 dataset. The reviews are categorized into three classes: **Accept**, **Reject**, and **Borderline**. We used a combination of deep learning (LSTM with Attention) and traditional machine learning (SVM with TF-IDF features) models to analyze and classify the reviews.

## Required Libraries

The following Python libraries are used in this project:

- `json`, `pandas`, `os`, `numpy`: For data processing and manipulation.
- `sklearn`: For splitting data, computing class weights, and implementing SVM.
- `tensorflow.keras`: For building deep learning models, including LSTM with Attention.
- `matplotlib`: For plotting training and validation metrics.
- `tkinter`: For building a GUI application to interact with the model.

## Steps

### 1. Data Preprocessing
The data is extracted from the ICLR 2017 review dataset. Each review is analyzed to extract the sentiment based on its rating and confidence.


### 2. Text Tokenization
The reviews are tokenized and padded to ensure consistent input size for the model.

# 3. Model Training
MILAM Model with Attention
A deep learning model based on LSTM with Attention mechanism is built using Keras.

SVM Model with TF-IDF Features
Additionally, a traditional machine learning model using Support Vector Machines (SVM) with unigrams and bigrams extracted via TF-IDF vectorizer is used.

