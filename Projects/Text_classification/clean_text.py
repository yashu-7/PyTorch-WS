from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

def clean_text(text):
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'[^A-Za-z\s]','',text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = stopwords.words('english')
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_words)