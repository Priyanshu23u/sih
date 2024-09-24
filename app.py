import os
import pandas as pd
import re
import spacy
import dateparser
from flask import Flask, jsonify, send_from_directory, request
from pymongo import MongoClient
from flask_cors import CORS

# Load spaCy's English model
nlp = spacy.load('en_core_web_sm')

# Create Flask app, using absolute path to ensure static files are found correctly
flask_app = Flask(__name__, static_folder=os.path.join(os.getcwd(), 'SIH-24-1867/my-project/dist'), static_url_path='/')
CORS(flask_app)  # Allow CORS for frontend-backend communication

# MongoDB setup
try:
    client = MongoClient('mongodb+srv://Priyanshu23u:24681012@cluster0.fyvfy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
    client.admin.command('ping')  # Check connection
    print("MongoDB connection established successfully.")
except Exception as e:
    print(f"Failed to connect to MongoDB: {str(e)}")
    exit()  # Exit if connection fails

db = client['sih1']
collection = db['newscontents']

# Serve the React frontend
@flask_app.route('/')
def serve():
    """Serve the React build's index.html file"""
    # Debug print to ensure Flask is looking in the correct folder
    print(f"Serving from: {flask_app.static_folder}")
    return send_from_directory(flask_app.static_folder, 'index.html')

# Serve any static files or routes for the React frontend
@flask_app.route('/<path:path>')
def static_proxy(path):
    """Serve the file if it exists, or fallback to index.html for React routing"""
    if os.path.exists(f'{flask_app.static_folder}/{path}'):
        return send_from_directory(flask_app.static_folder, path)
    else:
        return send_from_directory(flask_app.static_folder, 'index.html')

# Handle 404 errors by serving the React frontend
@flask_app.errorhandler(404)
def not_found(e):
    """Fallback for React Router"""
    return send_from_directory(flask_app.static_folder, 'index.html')

# Non-Indian countries to filter out location
non_indian_countries = [
    'Vietnam', 'Myanmar', 'USA', 'China', 'Japan', 'UK', 'Germany', 'France',
    'Brazil', 'Canada', 'Russia', 'Australia', 'Pakistan', 'Sri Lanka', 'Nepal', 
    'Bangladesh', 'Thailand', 'Singapore', 'Indonesia', 'Mexico', 'South Korea'
]

def extract_location(text):
    """Extract the first two locations (GPE) using spaCy's NER and filter out non-Indian countries."""
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE' and ent.text not in non_indian_countries]
    return ', '.join(locations[:2]) if locations else None

def extract_first_date(text):
    """Extract the first date from the given text using regular expressions and dateparser."""
    date_pattern = r'\b(?:[A-Z][a-z]+ \d{1,2}(?:, \d{4})?|(?:\d{4}-\d{2}-\d{2}))\b'
    matches = re.findall(date_pattern, text)
    if matches:
        parsed_date = dateparser.parse(matches[0], languages=['en'])
        if parsed_date:
            return parsed_date.strftime('%Y-%m-%d')
    return None

def extract_date_time(text):
    """Extract date and time using the extract_first_date function."""
    return extract_first_date(text)

def extract_disaster_type(text, disaster_keywords):
    """Extract disaster type based on a predefined list."""
    disaster = [d for d in disaster_keywords if d in text.lower()]
    return disaster[0] if disaster else "Unknown"

def extract_short_description(text):
    """Extract a short description from the text, limited to two sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sentences[:2])

def process_df(df):
    """Apply extraction functions to the DataFrame."""
    disaster_keywords = [
        'earthquake', 'flood', 'hurricane', 'tornado', 'tsunami', 
        'volcano', 'cyclone', 'wildfire', 'landslide', 'avalanche',
        'drought', 'heatwave', 'blizzard', 'storm', 'typhoon', 
        'hailstorm', 'mudslide', 'sandstorm', 'tremor', 'aftershock'
    ]

    df['location'] = df['content'].apply(extract_location)
    df['date'] = df['content'].apply(extract_date_time)
    df['disaster_type'] = df['content'].apply(lambda x: extract_disaster_type(x, disaster_keywords))
    df['short_description'] = df['content'].apply(extract_short_description)
    
    return df[['location', 'date', 'disaster_type', 'short_description']]

# New API endpoint to accept and process Twitter data
@flask_app.route("/api/process_twitter_data", methods=["POST"])
def process_twitter_data():
    try:
        data = request.json
        tweets = data.get('tweets', [])
        
        if not tweets:
            return jsonify({"error": "No tweet data provided"}), 400

        df = pd.DataFrame(tweets)
        processed_df = process_df(df)
        
        # Fetch existing data from MongoDB
        mongo_data = list(collection.find({}, {"_id": 0, "content": 1}))
        final_data = mongo_data + processed_df.to_dict(orient='records')

        return jsonify(final_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API endpoint to fetch and process news content from MongoDB
@flask_app.route("/api/final_news", methods=["GET"])
def get_news():
    try:
        # Fetch data from MongoDB
        mongo_data = list(collection.find({}, {"_id": 0, "content": 1}))

        if not mongo_data:
            return jsonify({"error": "No data found"}), 404

        df = pd.DataFrame(mongo_data)

        if df.empty:
            return jsonify({"error": "No valid content to process"}), 400

        processed_df = process_df(df)
        return jsonify(processed_df.to_dict(orient='records')), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API endpoint to extract information from provided text
@flask_app.route("/api/extract_info", methods=["POST"])
def extract_info():
    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        df = pd.DataFrame([{'content': text}])
        processed_df = process_df(df)

        if processed_df.empty:
            return jsonify({"error": "Could not extract information"}), 400

        return jsonify(processed_df.to_dict(orient='records')[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Test MongoDB connection
@flask_app.route("/test_db", methods=["GET"])
def test_db():
    try:
        test_data = collection.find_one({}, {"_id": 0, "content": 1})
        return jsonify({"test_data": test_data}) if test_data else jsonify({"error": "No test data found"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == "__main__":
    flask_app.run(debug=True)
