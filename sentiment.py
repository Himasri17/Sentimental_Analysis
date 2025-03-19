from flask import Flask, request, jsonify
from flask import Flask, render_template
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from flask_cors import CORS  # To allow frontend requests

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load the dataset
df = pd.read_csv(r"C:\Users\himas\OneDrive\Desktop\project\mlproject\train.csv", encoding='latin1')
test = pd.read_csv(r"C:\Users\himas\OneDrive\Desktop\project\mlproject\test.csv", encoding='latin1')

# Concatenate training and testing datasets
df = pd.concat([df, test])

# Drop unnecessary columns
df.drop(columns=['textID', 'Time of Tweet', 'Age of User', 'Country', 'Population -2020', 
                 'Land Area (Km²)', 'Density (P/Km²)', 'selected_text'], axis=1, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Basic text preprocessing
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower().strip()  # Convert to lowercase
    return text

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Encode the target variable
encoder = LabelEncoder()
df['sentiment'] = encoder.fit_transform(df['sentiment'])

# Split the dataset
X = df['cleaned_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Logistic Regression
lr = LogisticRegression(C=1, n_jobs=-1, random_state=42)
lr.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
joblib.dump(lr, 'logistic_regression_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Prediction Function
def predict_sentiment(text):
    text = preprocess_text(text)
    text_vectorized = tfidf.transform([text])
    pred = lr.predict(text_vectorized)
    sentiment = encoder.inverse_transform(pred)[0]
    return sentiment

# Flask API Route for Sentiment Analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    sentiment = predict_sentiment(text)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
