# Fake News Detection System

This repository contains a simple and quick implementation of a **Fake News Detection System** using Python, Streamlit, and Machine Learning. The project includes the following components:

---

## 📂 Files in the Repository

### 1. `main.py`
- This script is used to **build and train the model** for detecting fake news.
- It processes the dataset (`Fake.csv` and `True.csv`) and saves the trained model and TF-IDF vectorizer as `.pkl` files for later use.

### 2. `app.py`
- This is the **execution script** that runs the **Streamlit app**.
- It provides a user interface for testing the model.
- Users can input a news article, and the app will classify it as **Fake** or **Real**.

### 3. Datasets
- **`Fake.csv`**: Contains labeled data for fake news articles.
- **`True.csv`**: Contains labeled data for true news articles.

---

## 🚀 How to Use

### Step 1: Build the Model
1. Run `main.py` to process the dataset and train the model.
2. This will create the following files:
   - `model.pkl`: The trained logistic regression model.
   - `tfidf.pkl`: The TF-IDF vectorizer used to preprocess text.

### Step 2: Run the Streamlit App
1. Ensure `model.pkl` and `tfidf.pkl` are in the same directory as `app.py`.
2. Install Streamlit if not already installed:
   ```bash
   pip install streamlit
3. Run the App.py 

### Step 3: Dataset link: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets?resource=download
