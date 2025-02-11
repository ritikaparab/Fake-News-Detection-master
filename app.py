import streamlit as st
import pickle

# Load the trained model and TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Streamlit app
st.title("Fake News Detection System")
st.write("Enter a news article below to determine if it's **Real** or **Fake**.")

# Input field for the news article
news_text = st.text_area("News Article", height=200)

# Button to predict
if st.button("Check News"):
    if news_text.strip():
        # Preprocess the input text
        vectorized_text = tfidf.transform([news_text])
        
        # Make prediction
        prediction = model.predict(vectorized_text)[0]
        
        # Display result
        if prediction == 1:
            st.success("The news article is classified as **Real**.")
        else:
            st.error("The news article is classified as **Fake**.")
    else:
        st.warning("Please enter a news article to check.")

# Footer
st.write("---")

