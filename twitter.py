from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image
from io import BytesIO
import pandas as pd

# Connect to MongoDB
client = MongoClient('mongodb+srv://Priyanshu23u:24681012@cluster0.fyvfy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['sih1']
collection = db['tweets']

# Extract data from MongoDB
tweets_data = list(collection.find())

# Convert the extracted data into a DataFrame
df = pd.DataFrame(tweets_data)

# Combine 'createdAt' and 'fullText' into a new column 'content'
df['content'] = df['createdAt'].astype(str) + ': ' + df['fullText']

# Drop the original 'createdAt' and 'fullText' columns
df.drop(columns=['createdAt', 'fullText'], inplace=True)

# Function to extract text from a webpage
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text().strip()
    except Exception as e:
        print(f"Error extracting text from URL {url}: {e}")
        return ""

# Function to extract text from an image using OCR
def extract_image_text(image_url):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from image URL {image_url}: {e}")
        return ""

# Function to check and extract content from image or video URLs
def process_media_url(url):
    if any(ext in url for ext in ['.jpg', '.jpeg', '.png']):
        return extract_image_text(url)
    if any(ext in url for ext in ['.mp4', '.mov', 'video']):
        return "Extracted video content (placeholder)"
    return ""

# Extract content from URLs if present and merge into 'content'
for index, row in df.iterrows():
    url = row.get('url', '')
    if url:
        text_content = extract_text_from_url(url)
        media_content = process_media_url(url)
        
        df.at[index, 'content'] += f"\nExtracted Web Content: {text_content}"
        if media_content:
            df.at[index, 'content'] += f"\nExtracted Media Content: {media_content}"

# Prepare the DataFrame to send to Flask
tweets_json = df.to_dict(orient='records')

# Send the processed tweet data to the Flask app API
try:
    response = requests.post('http://localhost:5000/api/process_twitter_data', json={'tweets': tweets_json})
    if response.status_code == 200:
        print("Data successfully sent and processed:", response.json())
    else:
        print(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")
except Exception as e:
    print(f"Error occurred while sending data to the Flask app: {str(e)}")
