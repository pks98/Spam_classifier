import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

stem = PorterStemmer()

tf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


def transform(text):
    text = text.lower()  # Converting the text to lower case
    text = nltk.word_tokenize(text)  # Tokenizing the words

    temp = []
    for i in text:  # only alphanumeric words
        if i.isalnum():
            temp.append(i)

    text = temp[:]

    temp = []  # removing stop words and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            temp.append(i)

    text = temp[:]

    temp = []
    for i in text:
        temp.append(stem.stem(i))

    return " ".join(temp)


st.title('Spam classifier')

# preprocessing
sms = st.text_area('Enter the message')

if st.button('Predict'):
    
    transformed_sms = transform(sms)

    # vectorize
    vector_ip = tf.transform([transformed_sms])

    # predict
    res = model.predict(vector_ip)[0]

    # Display

    if res == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
