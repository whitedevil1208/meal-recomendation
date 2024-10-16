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
@app.post("/rec")
def get_rec(prompt:str):
    extracted_entities = extract_entities(prompt)
    print(f"Extracted Entities: {extracted_entities}")
    recommendation = recommend_food_based_on_entities(extracted_entities)
    print("Recommended Food:", recommendation)
    return {"rec":recommendation}


