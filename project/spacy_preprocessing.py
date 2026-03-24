import spacy
import pandas as pd


class SpacyPreprocessing:
    def __init__(self):
        self.nlp = spacy.load(
            "en_core_web_sm",
        )

    def preprocess_all(self, texts_list):
        """
        Приводит к нижнему регистру, лемматизирует,
        убирает стоп-слова, пунктуацию и лишние пробелы
        """
        processed_docs = []
        for doc in self.nlp.pipe(texts_list):
            lemmas = [
                token.lemma_.lower() for token in doc
                if not token.is_stop and not token.is_punct and not token.is_space
            ]

            processed_docs.append(' '.join(lemmas))

        return processed_docs


prep = SpacyPreprocessing()

df = pd.read_csv('wine_data/winemag-data-130k-v2.csv')
df_copy = df.sample(n=10000, random_state=42).copy()
df_copy['description_clean'] = prep.preprocess_all(df_copy['description'].astype(str).tolist())
df_copy.to_csv('wine_data/winemag-data-preprocessed.csv', index=False)
