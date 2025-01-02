import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure nltk `punkt` tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='./nltk_data')

nltk.data.path.append('./nltk_data')

# Load pre-trained model and vectorizer
MODEL_PATH = './model.pkl'
VECTORIZER_PATH = './vectorizer.pkl'

try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
        tfidf = pickle.load(vectorizer_file)
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

# Text transformation function
def transform_text(text):
    # Lowercase the text
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    # Remove stopwords (optional, add your own stopword list if needed)
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwords]
    # Return processed text as a single string
    return ' '.join(tokens)

# Streamlit app
st.title("Spam Detection App")
st.write("Enter a message below to determine if it's spam or not.")

# Input text
input_sms = st.text_input("Enter a message:")

# Predict button
if st.button("Predict"):
    if input_sms.strip():
        # Transform text
        transformed_sms = transform_text(input_sms)
        
        # Vectorize the transformed text
        vectorized_sms = tfidf.transform([transformed_sms])
        
        # Make prediction
        prediction = model.predict(vectorized_sms)[0]
        prediction_proba = model.predict_proba(vectorized_sms)[0]
        
        # Display result
        if prediction == 1:
            st.error("This message is classified as **SPAM**.")
            st.write(f"Confidence: {prediction_proba[1]:.2f}")
        else:
            st.success("This message is classified as **NOT SPAM**.")
            st.write(f"Confidence: {prediction_proba[0]:.2f}")
    else:
        st.warning("Please enter a message to analyze.")

# Footer
st.write("---")
st.write("Powered by [Your Name]")





