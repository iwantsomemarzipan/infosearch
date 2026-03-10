import numpy as np
import pandas as pd
import math


class ManualIndices:
    def __init__(self, k1=1.5, b=0.75):
        # константы для формулы bm-25
        self.k1 = k1
        self.b = b

    def build_vocab(self, texts):
        """Строит словарь уникальных слов корпуса для матричных индексов"""
        all_words = sorted(set(w for t in texts for w in t.split()))
        vocab = {word: i for i, word in enumerate(all_words)}
        return vocab

    def build_index(self, ids, texts, index_func):
        """
        Строит индекс с использованием выбранной функции
        """
        if index_func.__name__ in ['create_freq_matrix', 'create_bm25_matrix']:
            vocab = self.build_vocab(texts)
            return index_func(texts, vocab)
        else:
            return index_func(ids, texts)

    def build_tf_matrix(self, texts, vocab):
        """Строит term frequency матрицу"""
        num_docs = len(texts)
        num_words = len(vocab)

        tf_matrix = np.zeros((num_words, num_docs))

        for d_idx, text in enumerate(texts):
            words = text.split()

            for word in words:
                if word in vocab:
                    tf_matrix[vocab[word], d_idx] += 1

        return tf_matrix

    # пункт 6
    def create_freq_dict(self, ids, texts):
        """Строит частотный индекс через словарь"""
        freq_index = {}
        for doc_id, text in zip(ids, texts):
            word_count = {}
            words = str(text).split()

            for word in words:
                word_count[word] = word_count.get(word, 0) + 1

            for word, count in word_count.items():
                if word not in freq_index:
                    freq_index[word] = {}
                freq_index[word][doc_id] = count

        return freq_index

    # пункт 6
    def create_bm25_dict(self, ids, texts):
        """Строит bm-25 индекс через словарь"""
        # словарь длин документов
        doc_lengths = {i: len(t.split()) for i, t in zip(ids, texts)}
        # средняя длина документа
        avgdl = sum(doc_lengths.values()) / len(ids)

        # считаем term frequency
        freq_index = self.create_freq_dict(ids, texts)

        bm25_idx = {}
        N = len(ids)

        for word, docs in freq_index.items():
            # число документов, содержащих терм q
            n_q = len(docs)
            # idf для каждого слова
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)

            bm25_idx[word] = {}
            for doc_id, freq in docs.items():
                L_d = doc_lengths[doc_id]   # длина документа

                # итоговый скор
                score = idf * (freq * (self.k1 + 1)) / (
                    freq + self.k1 * (1 - self.b + self.b * (L_d / avgdl))
                )

                bm25_idx[word][doc_id] = score

        return bm25_idx

    # пункт 7
    def create_freq_matrix(self, texts, vocab):
        """Строит частотный индекс через матрицу"""
        return self.build_tf_matrix(texts, vocab)

    # пункт 7
    def create_bm25_matrix(self, texts, vocab):
        """Строит bm-25 индекс через матрицу"""
        tf_matrix = self.build_tf_matrix(texts, vocab)

        num_docs = tf_matrix.shape[1]
        L_d = tf_matrix.sum(axis=0)    # длины документов
        
        # средняя длина документа
        avgdl = np.mean(L_d)
        # число документов, содержащих терм q
        n_q = np.count_nonzero(tf_matrix, axis=1)
        # idf для каждого слова
        idf = np.log((num_docs - n_q + 0.5) / (n_q + 0.5) + 1).reshape(-1, 1)

        # считаем знаменатель
        denominator_part = (1 - self.b + self.b * (L_d / avgdl))
        denominator = tf_matrix + self.k1 * denominator_part
        # итоговая матрица
        bm25 = idf * (tf_matrix * (self.k1 + 1) / denominator)

        return bm25


df = pd.read_csv('reviews_preprocessed.csv')
texts = df['cleaned_text']

builder = ManualIndices()

index = builder.build_index(df.index, texts, builder.create_bm25_matrix)
print(index)
