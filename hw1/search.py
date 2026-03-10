from preprocessing import Preprocessing     # может долго подгружаться из-за spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bm25_vectorizer import BM25Vectorizer
import pandas as pd
import numpy as np


class SearchReviews:
    def __init__(self, preprocessing_class):
        self.prep = preprocessing_class
        self.count_vectorizer = CountVectorizer()
        self.bm25_vectorizer = BM25Vectorizer()

        self.df = None
        self.count_matrix = None
        self.bm25_matrix = None

    def fit(self, df, texts):
        """Строит индексы для корпуса текстов"""
        self.df = df
        self.count_matrix = self.count_vectorizer.fit_transform(texts)
        self.bm25_matrix = self.bm25_vectorizer.fit_transform(texts)

    def search(self, query, mode='count'):
        """Ищет по запросу для частотного или bm-25 индекса"""
        if mode == 'count':
            vectorizer = self.count_vectorizer
            matrix = self.count_matrix
        elif mode == 'bm25':
            vectorizer = self.bm25_vectorizer
            matrix = self.bm25_matrix

        # предобрабатываем запрос
        query_vector = vectorizer.transform(self.prep.preprocess_all([query]))

        similarity = cosine_similarity(matrix, query_vector)
        ranked_indices = np.argsort(similarity.flatten())[::-1]

        # выводим топ-5 документов
        return self.df.loc[ranked_indices].head(5)
    

prep = Preprocessing()
search = SearchReviews(preprocessing_class=prep)

# строим индексы
df = pd.read_csv('reviews_preprocessed.csv', index_col=0)
cleaned_texts = df['cleaned_text']
search.fit(df, cleaned_texts)

# поиск по запросу
res_count = search.search('пряный', mode='count')
res_bm25 = search.search('пряный', mode='bm25')

print(res_count)
print(res_bm25)
