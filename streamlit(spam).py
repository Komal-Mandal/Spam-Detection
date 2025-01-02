import streamlit as st
import pickle
import nltk
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('stopwords') 
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()



def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)

vectorizer = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

st.title("Email/sms classifier")

input_sms = st.text_input("Enter the message")

if st.button("predict"):
   transformed_sms = transform_text(input_sms)

   Vector_input = vectorizer.transform([transformed_sms])


   result = model.predict(Vector_input)[0]

   if result == 1:
      st.header("Spam")
   else:
     st.header("Not Spam")




