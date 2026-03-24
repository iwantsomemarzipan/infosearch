import argparse
import pandas as pd
from search_engine import SearchReviews
from query_preprocessing import QueryPreprocessing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['bm25', 'word2vec', 'fasttext'], default='bm25')

    args = parser.parse_args()

    prep = QueryPreprocessing()
    search = SearchReviews(prep)

    search.load_indexes('indexes')

    search.df = pd.read_csv('wine_data/winemag-data-preprocessed.csv')
    results = search.search(args.query, mode=args.mode)

    print(results[['title', 'description']])

if __name__ == "__main__":
    main()
