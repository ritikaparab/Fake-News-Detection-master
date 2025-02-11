import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load the datasets
fake_data = pd.read_csv('Fake.csv')  # Ensure the correct path to Fake.csv
true_data = pd.read_csv('True.csv')  # Ensure the correct path to True.csv

# Step 2: Add labels
fake_data['label'] = 0  # Label for fake news
true_data['label'] = 1  # Label for true news

# Step 3: Combine the datasets
combined_data = pd.concat([fake_data, true_data], ignore_index=True)

# Shuffle the combined dataset
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 4: Data Preprocessing
# Ensure necessary columns exist
if 'text' not in combined_data.columns:
    raise ValueError("Ensure that the column containing the news articles is named 'text'.")

# Keep only text and label columns
combined_data = combined_data[['text', 'label']]

# Drop missing values
combined_data = combined_data.dropna()

# Step 5: Feature Extraction
X = combined_data['text']
y = combined_data['label']

# Convert text to numerical features using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 7: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 9: Save the Model and TF-IDF Vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

print("Model and vectorizer saved successfully!")
