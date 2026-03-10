import spacy
import string
import pandas as pd


class Preprocessing:
    _translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    def __init__(self):
        self.nlp = spacy.load(
            "ru_core_news_sm",
            disable=["ner", "parser", "tok2vec", "attribute_ruler"]
        )

    @classmethod
    def _clean_text(cls, text):
        """Убирает пунктуацию и приводит к нижнему регистру"""
        text = text.translate(cls._translator).lower()
        return text.strip()

    def preprocess_all(self, texts_list):
        """Лемматизирует, убирает стоп-слова и лишние пробелы"""
        clean_texts = (self._clean_text(t) for t in texts_list)
        
        processed_docs = []
        for doc in self.nlp.pipe(clean_texts):
            lemmas = [
                token.lemma_ for token in doc
                if not token.is_stop and not token.is_space
            ]

            processed_docs.append(' '.join(lemmas))

        return processed_docs


prep = Preprocessing()

df = pd.read_csv('reviews.csv')
df['cleaned_text'] = prep.preprocess_all(df['review_text'].astype(str).tolist())

df.to_csv('reviews_preprocessed.csv', index=False)
