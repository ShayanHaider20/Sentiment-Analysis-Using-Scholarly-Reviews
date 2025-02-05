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

```python
def read_reviews(folder_path):
    reviews = []
    positive = ['good', 'strong', 'accept', 'strongly accept', 'good paper', 'above']
    borderline = ['marginally', 'probably', 'maybe', 'neutral']
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, encoding="utf8") as f:
                data = json.load(f)
            paper_id = data['id']
            for review in data['reviews']:
                review_text = review['review']
                rating = review['rating']
                confidence = review['confidence']
                if any(pos in rating.lower() for pos in positive):
                    preliminary_decision = 1
                elif any(border in rating.lower() for border in borderline):
                    preliminary_decision = 2
                else:
                    preliminary_decision = 0
                reviews.append({
                    'paper_id': paper_id,
                    'text': review_text,
                    'rating': rating,
                    'confidence': confidence,
                    'preliminary_decision': preliminary_decision
                })
    return reviews



tokenizer = Tokenizer(num_words=10000)
texts = df['text'].tolist()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=300)

