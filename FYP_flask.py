from flask import Flask, request, render_template, send_file
import numpy as np
import tensorflow as tf
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from werkzeug.exceptions import BadRequestKeyError
from sklearn.exceptions import NotFittedError
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the trained Keras model
model = tf.keras.models.load_model('saved_model')

# Load the TF-IDF vectorizer and LSA model
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('lsa_model.pkl', 'rb') as f:
    lsa = pickle.load(f)

stop_words = set(stopwords.words('english'))

# Configuring SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///news.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database model to store news
class News(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # Primary key
    title = db.Column(db.String(300), nullable=False)
    text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)  # Add a column for prediction (e.g., 'True' or 'Fake')

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Lowercase the text
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\d', '', text)  # Remove digits
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        title_data = request.form.get('title', '')  # Access the title field, default to empty string if not found
        text_data = request.form.get('text', '')  # Access the text field, default to empty string if not found
        if not text_data.strip():
            return render_template('error.html', error_message='Error: Text field is required.')

    except BadRequestKeyError as e:
        return render_template('error.html', error_message='Error: Invalid request.')

    # Combine title and text for prediction
    news_data = f"{title_data} {text_data}"
    
    # Continue with preprocessing and prediction
    processed_text = preprocess_text(news_data)
    
    # Transform text using TF-IDF and LSA
    try:
        text_vector = vectorizer.transform([processed_text])
    except NotFittedError as e:
        return render_template('error.html', error_message='Error: The vectorizer is not fitted.')

    text_lsa = lsa.transform(text_vector)
    
    # Predict with the Keras model
    prediction_prob = model.predict(text_lsa)
    prediction = (prediction_prob > 0.5).astype(int).flatten()[0]
    
    # Assign prediction text
    prediction_text = 'True' if prediction == 1 else 'Fake'
    
    # Save the result to the database
    new_entry = News(title=title_data, text=text_data, prediction=prediction_text)
    db.session.add(new_entry)
    db.session.commit()
    
    # Define the correct URL for verified news
    correct_url = "https://www.bbc.com/news"  # Example URL to redirect for verified news
    
    # Return the prediction result on a new page
    return render_template('results.html', prediction_text=f'The news is likely {prediction_text}.', correct_url=correct_url)

@app.route('/reliable-news')
def reliable_sources():
    return render_template('newsLink.html')


@app.route('/export')
def export_to_excel():
    try:
        # Query all records from the News table
        all_news = News.query.all()
        
        # Convert the records into a list of dictionaries
        news_data = [{
            'ID': news.id,
            'Title': news.title,
            'Text': news.text,
            'Prediction': news.prediction
        } for news in all_news]
        
        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(news_data)
        
        # Specify the Excel file path
        file_path = 'news_records.xlsx'
        
        # Export the DataFrame to an Excel file
        df.to_excel(file_path, index=False)
        
        # Send the file to the user
        return send_file(file_path, as_attachment=True)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database and tables
    app.run(debug=True)
