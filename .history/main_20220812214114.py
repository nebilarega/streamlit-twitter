import datetime as dt
import re
import pandas as pd
import streamlit as st
from sklean.feature_extraction.text import TfidfVectorizer


# Set page title
st.title('Twitter Sentiment Analysis')

# Load classification model
with st.spinner('Loading classification model...'):
    classifier = TextClassifier.load('models/best-model.pt')

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
    sentence = Sentence(preprocess(tweet_input))

    # Make predictions
    with st.spinner('Predicting...'):
        classifier.predict(sentence)

    # Show predictions
    label_dict = {'0': 'Negative', '4': 'Positive'}

    if len(sentence.labels) > 0:
        st.write('Prediction:')
        st.write(label_dict[sentence.labels[0].value] + ' with ',
                sentence.labels[0].score*100, '% confidence')

try:
    st.write(tweet_data)
    try:
        st.write('Positive to negative tweet ratio:', pos_vs_neg['4']/pos_vs_neg['0'])
    except ZeroDivisionError: # if no negative tweets
        st.write('All postive tweets')
except NameError: # if no queries have been made yet
    pass