import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_input(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Load the trained model
filename_model = 'logistic_regression_model.pkl'
filename_vectorizer = 'tfidf_vectorizer.pkl'

loaded_model = joblib.load(filename_model)
vectorizer = joblib.load(filename_vectorizer)

# Streamlit app
def main():
    # Set page title and favicon
    st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="logo.jpeg")

    # Set title and description
    st.title("Twitter Sentiment Analysis")
    st.write("This app classifies the sentiment of tweets as positive or negative.")

    # Input field for user to enter tweet
    user_input = st.text_input("Enter a tweet:")

    # Classify the input tweet when the user clicks the button
    if st.button("Classify"):
        if user_input:
            try:
                # Preprocess the user input
                preprocessed_tweet = preprocess_input(user_input)

                # Transform the preprocessed text using the loaded vectorizer
                tweet_tfidf = vectorizer.transform([preprocessed_tweet])

                # Predict the sentiment of the new text
                pred = loaded_model.predict(tweet_tfidf)

                # Print the prediction and sentiment with an emoji
                if pred[0] == 0:
                    st.write("Predicted Sentiment: Negative Tweet 😞")
                else:
                    st.write("Predicted Sentiment: Positive Tweet 😊")
            except NotFittedError:
                st.error("The TF-IDF vectorizer is not fitted. Please fit the vectorizer with your training data.")
        else:
            st.write("Please enter a tweet.")

if __name__ == "__main__":
    main()
