import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import openai


# Allow specific orig
# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the Datasets
df_normal = pd.read_excel(r"NLP DATASET.xlsx")
df_diseased = pd.read_excel(r"nlp dieseases dataset.xlsx")

# Preprocessing Function to clean text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\W+', ' ', text.lower())
    words = text.split()
    words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words if word not in stop_words]
    return ' '.join(words)

# Combine columns to create 'features' for both datasets
df_normal['features'] = (df_normal['type'] + " " + df_normal['meal type'] + " " + 
                         df_normal['person type'] + " " + df_normal['diet type'])
df_normal['features'] = df_normal['features'].apply(preprocess_text)

df_diseased['features'] = (df_diseased['type'] + " " + df_diseased['meal type'] + 
                           " " + df_diseased['disease type'])
df_diseased['features'] = df_diseased['features'].apply(preprocess_text)

# Fine-tuned NER model setup
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
ner_pipeline = pipeline("ner", model=model_name, tokenizer=model_name, aggregation_strategy="simple")

# Improved rule-based entity extraction
def rule_based_entity_extraction(text):
    meal_types = ['breakfast', 'lunch', 'dinner', 'snack']
    food_types = ['veg', 'non-veg', 'vegetarian', 'meat', 'chicken', 'fish', 'salmon', 'egg']
    person_types = ['child', 'adult', 'elderly', 'senior', 'athlete', 'pregnant','normal']
    diet_types = ['low-carb', 'high-protein', 'low-fat', 'vegan', 'keto', 'paleo']
    disease_types = ['diabetes', 'hypertension', 'obesity', 'heart disease']

    entities = {
        'meal_type': None,
        'food_type': None,
        'person_type': None,
        'diet_type': None,
        'disease_type': None
    }
    
    for meal in meal_types:
        if meal in text.lower():
            entities['meal_type'] = meal
            break
    
    for food in food_types:
        if food in text.lower():
            entities['food_type'] = food
            break
    
    for person in person_types:
        if person in text.lower():
            entities['person_type'] = person
            break
    
    for diet in diet_types:
        if diet in text.lower():
            entities['diet_type'] = diet
            break
    
    for disease in disease_types:
        if disease in text.lower():
            entities['disease_type'] = disease
            break
    
    return entities

# Extract entities using both rule-based and fine-tuned BERT-based NER
def extract_entities(text):
    bert_entities = ner_pipeline(text)
    rule_entities = rule_based_entity_extraction(text)
    
    entities = {
        'meal_type': rule_entities['meal_type'],
        'food_type': rule_entities['food_type'],
        'person_type': rule_entities['person_type'],
        'diet_type': rule_entities['diet_type'],
        'disease_type': rule_entities['disease_type'],
    }
    
    return entities

# Load a better pre-trained sentence transformer model
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to get sentence embedding
def get_sentence_embedding(text, model):
    return model.encode(text)

# Vectorize both normal and diseased datasets using sentence embeddings
df_normal['embedding'] = df_normal['features'].apply(lambda x: get_sentence_embedding(x, sentence_model))
df_diseased['embedding'] = df_diseased['features'].apply(lambda x: get_sentence_embedding(x, sentence_model))

# Function to recommend food based on extracted entities
def recommend_food_based_on_entities(entities):
    if entities['disease_type']:
        user_input = f"{entities['food_type']} {entities['meal_type']} {entities['disease_type']}"
        user_embedding = get_sentence_embedding(user_input, sentence_model)
        similarities = cosine_similarity([user_embedding], np.stack(df_diseased['embedding'].values))
        idx = np.argmax(similarities)
        return df_diseased['recommend'].iloc[idx]
    else:
        user_input = f"{entities['food_type']} {entities['meal_type']} {entities['person_type']} {entities['diet_type']}"
        user_embedding = get_sentence_embedding(user_input, sentence_model)
        similarities = cosine_similarity([user_embedding], np.stack(df_normal['embedding'].values))
        idx = np.argmax(similarities)
        return df_normal['recommend'].iloc[idx]

# Function to get user input and recommend food
def get_recommendation():
    prompt = input("Please describe your diet preferences: ")
    extracted_entities = extract_entities(prompt)
    print(f"Extracted Entities: {extracted_entities}")
    recommendation = recommend_food_based_on_entities(extracted_entities)
    print("Recommended Food:", recommendation)

# Run the system
app = FastAPI()

origins = [
    "http://localhost:5173",  # React app during development
    "https://your-frontend-domain.com",  # Production front-end domain
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, etc.)
    allow_headers=["*"],  # Allows all headers, or restrict to specific ones like 'Content-Type'
)


@app.post("/rec")
def get_rec(prompt:str):
    extracted_entities = extract_entities(prompt)
    print(f"Extracted Entities: {extracted_entities}")
    recommendation = recommend_food_based_on_entities(extracted_entities)
    print("Recommended Food:", recommendation)
    return {"rec":recommendation}

import requests

# Set your Edamam API credentials
EDAMAM_APP_ID = '2a98c6a4'  # Replace with your Edamam APP ID
EDAMAM_APP_KEY = '75704984fc122dc3153ae7a943f3cb56'  # Replace with your Edamam APP KEY

# Function to get nutritional information from Edamam API
def get_nutritional_info(food_item):
    url = f"https://api.edamam.com/api/nutrition-data?app_id={EDAMAM_APP_ID}&app_key={EDAMAM_APP_KEY}&nutrition-type=logging&ingr={food_item}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to format nutritional information for output
def format_nutrition_info(nutrition_data):
    if nutrition_data:
        nutrients = nutrition_data.get('totalNutrients', {})
        formatted_info = []
        for nutrient, details in nutrients.items():
            formatted_info.append(f"{details['label']}: {details['quantity']} {details['unit']}")
        return "\n".join(formatted_info)
    return "Nutritional information not available."

# Function to get user input and perform nutritional analysis
@app.post("/nut")
def get_nutritional_analysis(food_item:str):
    # Get user input for food item

    # Get nutritional analysis for the food item
    nutrition = get_nutritional_info(food_item)
    nutritional_analysis = format_nutrition_info(nutrition)
    print("Nutritional Analysis:\n", nutritional_analysis)
    return {"res":nutritional_analysis}

openai.api_key = "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"

@app.post("/analyze")
async def analyze_input(user_input: str):
    prompt = user_input
    
    try:
        # Call OpenAI API to analyze user input
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        # Extract the response text
        openai_response = response.choices[0].message['content'].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error from OpenAI: {str(e)}")

    return {
        "analysis": openai_response
    }