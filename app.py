import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    file_path = 'data/netflix.csv'
    try:
        df = pd.read_csv(file_path, sep=',', encoding='latin1')

        print("Size of dataset:", df.shape)
        print("\nColomn info:")
        print(df.info())
        print("\nfirst 5 :")
        print(df.head())

        return df
    except Exception as e:
        print(f"Loading Error: {e}")
        return None

def preprocess_data(df):
    if df is None:
        return None
    df_unique = df.drop_duplicates()
    df_unique = df_unique.fillna('')

    selected_cols = ['title', 'type', 'listed_in', 'description']
    df_processed = df_unique[selected_cols]

    print("Preprocessed data shape:", df_processed.shape)
    return df_processed

def create_feature_matrix(df):
    if df is None:
        return None
    # Объединим все текстовые признаки в один
    df['features'] = df['type'] + ' ' + df['listed_in'] + ' ' + df['description']

    # Создание TF-IDF матрицы
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['features'])

    print("Feature matrix shape:", tfidf_matrix.shape)
    return tfidf_matrix

def calculate_similarity(feature_matrix):
    try:
        similarity_matrix = cosine_similarity(feature_matrix)
        print("similarity: ", similarity_matrix.shape)
        return similarity_matrix
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return None

def get_recomendation(title, df, similarity_matrix):

    """Args:
    title (str): Название фильма для поиска рекомендаций.
    df (pandas.DataFrame): DataFrame с информацией о фильмах.
    similarity_matrix (numpy.ndarray): Матрица сходства фильмов.

    Returns:
    list: Список рекомендованных фильмов или сообщение об ошибке.
    """
    try:
        movie_indices =  df[df['title'] == title].index
        if len(movie_indices) == 0:
            return "Movie not found"
        movie_index = movie_indices[0]

        similarity_scores = list(enumerate(similarity_matrix[movie_index]))
        similarity_scores = sorted(similarity_scores, key = lambda x:[1], reverse=True)
        similarity_scores = similarity_scores[1:11]

        recomendations = []
        for i in similarity_scores:
            movie_title= df.iloc[i]['title']
            movie_type=df.iloc[i]['type']
            recomendations.append(f"{movie_title} ({movie_type})")
            return recomendations
    except Exception as e:
        return f"Error in recomendatioans: {str(e)} "


# Вызов функции для проверки
df = load_data()
df_processed = preprocess_data(df)

if df_processed is not None:
    feature_matrix = create_feature_matrix(df_processed)
    if feature_matrix is not None:
        print("Feature matrix created successfully")

        # Вызов функции calculate_similarity
        similarity_matrix = calculate_similarity(feature_matrix)
        if similarity_matrix is not None:
            print("Similarity matrix created successfully")
        else:
            print("Failed to create similarity matrix")
    else:
        print("Failed to create feature matrix")
else:
    print("Preprocessing failed, cannot create feature matrix")
