import pandas as pd
from bm25_vectorizer import BM25Vectorizer
from search_engine import SearchReviews

df = pd.read_csv('wine_data/winemag-data-preprocessed.csv')
texts = df['description_clean'].tolist()

search = SearchReviews()

# строим индексы для BM25
search.bm25_vectorizer = BM25Vectorizer()
search.bm25_matrix = search.bm25_vectorizer.fit_transform(texts)

# строим индексы для w2v и fasttext
search.fit(df, texts)

# cохраняем полученные индексы
search.save_indexes('indexes')
