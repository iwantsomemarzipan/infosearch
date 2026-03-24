import numpy as np
import time
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity


class SearchReviews:
    def __init__(self, preprocessing_class=None):
        self.prep = preprocessing_class

        self.bm25_vectorizer = None

        self.w2v_model = None
        self.ft_model = None

        self.df = None

        self.bm25_matrix = None
        self.w2v_matrix = None
        self.ft_matrix = None

    def _get_vector(self, text, model):
        """Вычисляет средний вектор текста"""
        words = str(text).split()
        vectors = [model[w] for w in words if w in model]

        if not vectors:
            return np.zeros(model.vector_size)

        return np.mean(vectors, axis=0)

    def fit(self, df, texts):
        """Переводит тексты в векторные матрицы"""
        self.df = df

        if self.w2v_model is None:
            from gensim.models import KeyedVectors
            self.w2v_model = KeyedVectors.load("models/word2vec.model", mmap='r')
        self.w2v_matrix = np.array([
            self._get_vector(t, self.w2v_model) for t in texts
        ])

        if self.ft_model is None:
            from gensim.models import KeyedVectors
            self.ft_model = KeyedVectors.load("models/fasttext.model", mmap='r')
        self.ft_matrix = np.array([
            self._get_vector(t, self.ft_model) for t in texts
        ])

    def search(self, query, mode='bm25', top_k=5):
        """Ищет документы по запросу"""
        start_time = time.time()
        clean_query = self.prep.preprocess_all([query])[0]

        if mode == 'bm25':
            query_vector = self.bm25_vectorizer.transform([clean_query])
            matrix = self.bm25_matrix

        elif mode == 'word2vec':
            if self.w2v_model is None:
                from gensim.models import KeyedVectors
                self.w2v_model = KeyedVectors.load("models/word2vec.model", mmap='r')
            query_vector = self._get_vector(clean_query, self.w2v_model).reshape(1, -1)
            matrix = self.w2v_matrix

        elif mode == 'fasttext':
            if self.ft_model is None:
                from gensim.models import KeyedVectors
                self.ft_model = KeyedVectors.load("models/fasttext.model", mmap='r')
            query_vector = self._get_vector(clean_query, self.ft_model).reshape(1, -1)
            matrix = self.ft_matrix

        similarity = cosine_similarity(matrix, query_vector).flatten()
        ranked_indices = np.argsort(similarity)[::-1][:top_k]

        elapsed_time = time.time() - start_time
        print(f"Search [{mode}] took: {elapsed_time:.4f} seconds")

        return self.df.iloc[ranked_indices]

    def save_indexes(self, index_folder='indexes'):
        """Сохраняет индексы"""
        os.makedirs(index_folder, exist_ok=True)

        np.save(f'{index_folder}/w2v_matrix.npy', self.w2v_matrix)
        np.save(f'{index_folder}/ft_matrix.npy', self.ft_matrix)

        with open(f'{index_folder}/bm25.pkl', 'wb') as f:
            pickle.dump({'vectorizer': self.bm25_vectorizer, 'matrix': self.bm25_matrix}, f)

        print(f"Indexes saved to '{index_folder}'")

    def load_indexes(self, folder='indexes'):
        """Загружает индексы"""
        self.w2v_matrix = np.load(f'{folder}/w2v_matrix.npy')
        self.ft_matrix = np.load(f'{folder}/ft_matrix.npy')

        with open(f'{folder}/bm25.pkl', 'rb') as f:
            data = pickle.load(f)
            self.bm25_vectorizer = data['vectorizer']
            self.bm25_matrix = data['matrix']
