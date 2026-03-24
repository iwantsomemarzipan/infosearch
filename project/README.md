# Описание проекта

## Корпус
В качестве корпуса был взят датасет [Wine Reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews?select=winemag-data-130k-v2.csv) c Kaggle. Предобработанный корпус содержит 10к вхождений из оригинального датасета.

## Модели
В качестве моделей были выбраны:
- **BM25** из библиотеки `bm25_vectorizer`
- **Word2vec** из библиотеки `gensim`
- **fastText** из библиотеки `gensim`

## Архитетура проекта
В корневой папке:
- `spacy_preprocessing.py`: скрипт, использовавшийся для предобработки датасета
- `download_models.py`: загружает модели Word2vec и fastText через API и сохраняет локально
- `search_engine.py`: содержит класс `SearchReviews`, который необходим для построения индексов и реализации поиска
- `build_indexes.py`: скрипт, где строились индексы
- `query_preprocessing.py`: содержит класс `QueryPreprocessing`, который используется для предобработки запроса
- `main.py`: CLI для поиска по запросу и готовым индексам

В папке `wine_data` лежит корпус с предобработанными текстами. Оригинальный датасет не загружен, так как файл слишком большой для гитхаба, но он для запуска поиска и не нужен.

В папке `indexes` лежат готовые индексы моделей.

## Запуск через консоль
Установка зависимостей и моделей

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
run download_models.py
```

Пример ввода поискового запроса

```
python main.py --query sweet red wine --mode bm25
```

Для аргумента `query` записывается текст запроса, а `mode` ставит нужную модель для поиска: bm25, word2vec, fasttext.

Также в `search_examples.ipynb` можно посмотреть на примеры запросов и выводы к ним.
