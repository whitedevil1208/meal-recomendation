from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure you download stopwords and punkt before use
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_excel(r"C:\Users\VIKRAM\Downloads\NLP DATASET.xlsx")  # Replace with your dataset path

# Step 1: Preprocessing Function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove numbers and special characters
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Join the cleaned words back into a single string
    return ' '.join(words)

# Apply preprocessing to each of the columns that we'll use
df['type'] = df['type'].apply(preprocess_text)
df['meal type'] = df['meal type'].apply(preprocess_text)
df['person type'] = df['person type'].apply(preprocess_text)
df['diet type'] = df['diet type'].apply(preprocess_text)

# Combine the relevant columns to create a feature set for the recommendation system
df['features'] = df['type'] + " " + df['meal type'] + " " + df['person type'] + " " + df['diet type']

# Initialize TF-IDF Vectorizer for feature extraction
vectorizer = TfidfVectorizer()

# Fit and transform the text data into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(df['features'])

# Define the recommendation function
def recommend_food_nlp(meal_type, food_type, person_type, diet_type):
    # Preprocess the user input
    food_type = preprocess_text(food_type)
    meal_type = preprocess_text(meal_type)
    person_type = preprocess_text(person_type)
    diet_type = preprocess_text(diet_type)

    # Combine the inputs into a query
    user_input = f"{food_type} {meal_type} {person_type} {diet_type}"

    # Transform the user input into the TF-IDF vector
    user_tfidf = vectorizer.transform([user_input])

    # Calculate the cosine similarity between user input and dataset
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)

    # Get the index of the highest similarity score
    idx = similarities.argsort()[0][-1]

    # Return the recommended food
    return df['recommend'].iloc[idx]

# Initialize the Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling the recommendation
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user inputs from the form
    meal_type = request.form['meal_type']
    food_type = request.form['food_type']
    person_type = request.form['person_type']
    diet_type = request.form['diet_type']

    # Get the recommendation based on user input
    recommendation = recommend_food_nlp(meal_type, food_type, person_type, diet_type)
    
    # Render the result on the webpage
    return render_template('result.html', recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
