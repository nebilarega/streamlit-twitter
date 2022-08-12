import datetime as dt
import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# Set page title
st.title('Twitter Sentiment Analysis')

# Load classification model
with st.spinner('Loading classification model...'):
    classifier_pol = joblib.load("text_classifier_polarity_model.joblib")
    classifier_sub = joblib.load("text_classifier_subjectivity_model.joblib")

# Preprocess function
allowed_chars = ' AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789~`!@#$%^&*()-=_+[]{}|;:",./<>?'
punct = '!?,.@#'
maxlen = 280

def preprocess_text(text):
    # Delete URLs, cut to maxlen, space out punction with spaces, and remove unallowed chars
    re.sub(r'http\S+', '', text) 

    text = re.sub(r'\W', ' ', str(text))

    # remove all single characters
    text= re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 

    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # Removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text)

    # Converting to Lowercase
    text = text.lower()

    return text

# def preprocess(text):
#     # Delete URLs, cut to maxlen, space out punction with spaces, and remove unallowed chars
#     return ''.join([' ' + char + ' ' if char in punct else char for char in [char for char in re.sub(r'http\S+', 'http', text, flags=re.MULTILINE) if char in allowed_chars]])

### SINGLE TWEET CLASSIFICATION ###
st.subheader('Single tweet classification')

# Get sentence input, preprocess it, and convert to flair.data.Sentence format
tweet_input = st.text_input('Tweet:')

if tweet_input != '':
    # Pre-process tweet
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3), min_df=1, max_df=1.0, strip_accents='unicode', lowercase=True, use_idf=True, smooth_idf=True, sublinear_tf=True)
    procecssed_feature_vectorized = vectorizer.fit_transform([preprocess_text(tweet_input)])

    # Make predictions
    with st.spinner('Predicting...'):
        polarity_prediction = classifier_pol.predict(procecssed_feature_vectorized)
        subjectivity_prediction = classifier_sub.predict(procecssed_feature_vectorized)

    # Show predictions
    label_dict = {'0': 'Negative', '4': 'Positive'}
    st.write('Polarity:', label_dict[str(polarity_prediction[0])])
    st.write('Subjectivity:', label_dict[str(subjectivity_prediction[0])])   