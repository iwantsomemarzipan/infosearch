import gensim.downloader as api
import os

os.makedirs("models", exist_ok=True)

print("Loading Word2Vec...")
w2v = api.load("word2vec-google-news-300")
w2v.save("models/word2vec.model")

print("Loading FastText...")
ft = api.load("fasttext-wiki-news-subwords-300")
ft.save("models/fasttext.model")

print("Done.")
