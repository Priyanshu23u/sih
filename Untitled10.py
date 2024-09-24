#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


import nltk
from nltk.corpus import stopwords
import re


# In[2]:


train_df = pd.read_csv('ans1.csv')
test_df = pd.read_csv('news_data1.csv')


# In[ ]:





# In[3]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train = train_df['Title'].values
y_train = train_df['label'].values
X_test = test_df['Title'].values

# Tokenization and Padding
max_words = 10000  # Max vocabulary size
max_len = 20       # Max sequence length

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# CNN Model
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    Conv1D(128, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    Conv1D(64, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2)
predictions = model.predict(X_test_pad)
predicted_labels = (predictions > 0.5).astype(int) 


# In[ ]:





# In[4]:


import pandas as pd
from nltk.corpus import stopwords
import nltk
import pickle
# Ensure stopwords are downloaded
nltk.download('stopwords')

word_index = tokenizer.word_index

stop_words = set(stopwords.words('english'))

filtered_word_index = {word: index for word, index in word_index.items() if word not in stop_words}

word_freq_df = pd.DataFrame(list(filtered_word_index.items()), columns=['Word', 'Index'])

most_common_words = word_freq_df.sort_values(by='Index').head(20)  # Fetch top 20 common words

print(most_common_words)


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Vectorize the text (Bag of Words)
vectorizer = CountVectorizer(max_features=5000)
X_train_vec_split = vectorizer.fit_transform(X_train_split)
X_val_vec_split = vectorizer.transform(X_val_split)

# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train_vec_split, y_train_split)

lr_val_predictions = lr_model.predict(X_val_vec_split)

accuracy = accuracy_score(y_val_split, lr_val_predictions)
print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")


# In[6]:


from sklearn.svm import SVC

# SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vec_split, y_train_split)

svm_val_predictions = svm_model.predict(X_val_vec_split)

accuracy = accuracy_score(y_val_split, svm_val_predictions)
print(f"SVM Accuracy: {accuracy * 100:.2f}%")


# In[7]:


from sklearn.ensemble import RandomForestClassifier

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_vec_split, y_train_split)

rf_val_predictions = rf_model.predict(X_val_vec_split)

accuracy = accuracy_score(y_val_split, rf_val_predictions)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")


# In[9]:


pickle.dump(rf_model,open("model.pkl","wb"))


# In[8]:


# Remove duplicates
test_df = test_df.drop_duplicates()

# Filter rows where the predicted label is 1
# = test_df[test_df['label'] == 1]
test_df = pd.read_csv('ans1.csv')
test_df_filtered = test_df[test_df['label'] == 1]
test_df_filtered.to_csv('ans2.csv', index=False)


# In[9]:


test_df_filtered


# In[10]:


import pymongo
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb+srv://Priyanshu23u:24681012@cluster0.fyvfy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')  # Adjust connection string as needed

# Access the database and collection
db = client['sih1']
collection = db['news1']

# Remove previous records from the collection
delete_result = collection.delete_many({})  # Deletes all documents in the collection
print(f"Deleted {delete_result.deleted_count} records from 'news1' collection.")

# Assuming 'test_df_filtered' is your DataFrame that you want to insert
data_dict = test_df_filtered.to_dict("records")  # Convert DataFrame to list of dictionaries

# Insert the new data into the collection
insert_doc = collection.insert_many(data_dict)

# Print inserted IDs (list of document IDs)
print(f"Inserted {len(insert_doc.inserted_ids)} new records.")
print(f"Inserted IDs: {insert_doc.inserted_ids}")

# Close the connection
client.close()


# In[11]:


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
    # Check for image content
    if any(ext in url for ext in ['.jpg', '.jpeg', '.png']):
        return extract_image_text(url)
    
    # Placeholder for video content extraction
    if any(ext in url for ext in ['.mp4', '.mov', 'video']):
        return "Extracted video content (placeholder)"
    
    return ""

# Extract content from URLs if present and merge into 'content'
for index, row in df.iterrows():
    url = row.get('url', '')
    if url:
        text_content = extract_text_from_url(url)
        media_content = process_media_url(url)
        
        # Merge extracted web and media content into 'content' column
        df.at[index, 'content'] += f"\nExtracted Web Content: {text_content}"
        if media_content:
            df.at[index, 'content'] += f"\nExtracted Media Content: {media_content}"

# Display the final DataFrame
print(df)



print("Data has been successfully processed and inserted into the 'tweetcontents' collection.")


# In[12]:


# Import necessary libraries
from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB



# In[ ]:





# In[ ]:





# In[13]:


import pandas as pd
import re
import dateparser
import spacy
import datefinder

# Load spaCy's English model
nlp = spacy.load('en_core_web_sm')

def extract_location(text):
    """Extract the first two locations (GPE) using spaCy's NER."""
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    
    # Return only the first two locations if available
    return ', '.join(locations[:2]) if locations else None

def extract_date(text):
    """Extract date using datefinder."""
    date_matches = list(datefinder.find_dates(text, index=True))
    if date_matches:
        return date_matches[0][0].strftime('%Y-%m-%d')
    return None

def extract_date_time(text):
    """Extract date and time using separate functions."""
    date_str = extract_date(text)
    return date_str

def extract_disaster_type(text, disaster_keywords):
    """Extract disaster type based on a predefined list."""
    disaster = [disaster for disaster in disaster_keywords if disaster in text.lower()]
    return disaster[0] if disaster else "Unknown"

def extract_short_description(text):
    """Extract a short description from the text, limited to two lines."""
    # Split text into sentences and keep only the first two
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    short_desc = ' '.join(sentences[:2])  # Join the first two sentences
    return short_desc

def process_df(df):
    """Apply extraction functions to the DataFrame."""
    disaster_keywords = [
        'earthquake', 'flood', 'hurricane', 'tornado', 'tsunami', 
        'volcano', 'cyclone', 'wildfire', 'landslide', 'avalanche',
        'drought', 'heatwave', 'blizzard', 'storm', 'typhoon', 
        'hailstorm', 'mudslide', 'sandstorm', 'tremor', 'aftershock'
    ]  # Example disaster types

    df['location'] = df['content'].apply(extract_location)  # Extract first two locations
    df['date'] = df['content'].apply(extract_date_time)  # Extract date
    df['disaster_type'] = df['content'].apply(lambda x: extract_disaster_type(x, disaster_keywords))  # Extract disaster type
    df['short_description'] = df['content'].apply(extract_short_description)  # Extract short description (2 lines)
    
    # Drop the 'date_time' column if it exists
    if 'date_time' in df.columns:
        df = df.drop(columns=['date_time'])
    
    return df
# Example usage:
# df = pd.read_csv('your_data.csv')  # Assuming your data is in CSV format
# processed_df = process_df(df)
# print(processed_df.head())


# In[14]:


processed_df = process_df(df)
print(processed_df)


# In[15]:


import pymongo
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb+srv://Priyanshu23u:24681012@cluster0.fyvfy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')

# Access the database and collection
db = client['sih1']  # Use your database name
collection = db['final_tweet']  # Use your collection name

# Remove previous records from the collection
delete_result = collection.delete_many({})  # Deletes all documents in the collection
print(f"Deleted {delete_result.deleted_count} records from 'final_tweet' collection.")

# Data to insert (Example)
data_dict = processed_df.to_dict("records")  # Assuming df_processed is the DataFrame

# Insert the new data into the collection
insert_doc = collection.insert_many(data_dict)

# Print inserted IDs (list of document IDs)
print(f"Inserted {len(insert_doc.inserted_ids)} new records.")
print(f"Inserted IDs: {insert_doc.inserted_ids}")

# Close the MongoDB connection
client.close()


# In[16]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb+srv://Priyanshu23u:24681012@cluster0.fyvfy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['sih1']
collection = db['newscontents']

news_data = list(collection.find())

# Convert the extracted data into a DataFrame
df1 = pd.DataFrame(news_data)


# Display the DataFrame
print(df1)
print("Data has been successfully inserted into the 'newscontents' collection.")


# In[17]:


processed_df1 = process_df(df1)
print(processed_df1)


# In[18]:


import pymongo
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb+srv://Priyanshu23u:24681012@cluster0.fyvfy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')  # Adjust connection string as needed

# Access the database and collection
db = client['sih1']  # Use your database name
collection = db['final_news']  # Use your collection name

# Remove previous records from the collection
delete_result = collection.delete_many({})  # Deletes all documents in the collection
print(f"Deleted {delete_result.deleted_count} records from 'final_news' collection.")
# Data to insert (Example)
data_dict =processed_df1.to_dict("records")

# Insert the new data into the collection
insert_doc = collection.insert_many(data_dict)

# Print inserted IDs (list of document IDs)
print(f"Inserted IDs: {insert_doc.inserted_ids}")

# Close the MongoDB connection
client.close()


# In[19]:


processed_df1.to_csv('processed_data.csv', index=False, date_format='%Y-%m-%d')


# In[20]:


df1


# In[21]:


processed_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




