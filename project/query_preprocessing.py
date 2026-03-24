from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re


class QueryPreprocessing:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            import nltk
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))

    def preprocess_all(self, texts_list):
        processed = []
        for text in texts_list:
            text = text.lower()
            text = re.sub(r"[^a-z\s]", " ", text)
            
            words = text.split()
            words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]

            processed.append(" ".join(words))
        return processed
