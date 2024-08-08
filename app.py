import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    # Loads the Netflix dataset from a CSV file and returns a pandas DataFrame
    file_path = 'data/netflix.csv'
    try:
        df = pd.read_csv(file_path, sep=',', encoding='latin1')

        print("Size of dataset:", df.shape)
        print("\nColumn info:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())

        return df
    except Exception as e:
        print(f"Loading Error: {e}")
        return None

def preprocess_data(df):
    # Preprocesses the input DataFrame by removing duplicates and handling missing values
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
    # Combine all text features into one
    df['features'] = df['type'] + ' ' + df['listed_in'] + ' ' + df['description']

    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['features'])

    print("Feature matrix shape:", tfidf_matrix.shape)
    return tfidf_matrix

def calculate_similarity(feature_matrix):
    # Calculates the cosine similarity matrix for the given feature matrix
    try:
        similarity_matrix = cosine_similarity(feature_matrix)
        print("Similarity matrix shape:", similarity_matrix.shape)
        return similarity_matrix
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return None

def get_recommendation(title, df, similarity_matrix):
    # Finds and returns a list of recommended movies based on the given movie title
    try:
        movie_indices = df[df['title'] == title].index
        if len(movie_indices) == 0:
            return ["Movie not found"]

        movie_index = movie_indices[0]

        similarity_scores = list(enumerate(similarity_matrix[movie_index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:11]

        recommendations = []
        for i, score in similarity_scores:
            movie_title = df.iloc[i]['title']
            movie_type = df.iloc[i]['type']
            recommendations.append(f"{movie_title} ({movie_type})")

        return recommendations
    except Exception as e:
        return [f"Error in recommendations: {str(e)}"]

# Call the function to check
df = load_data()
df_processed = preprocess_data(df)

if df_processed is not None:
    feature_matrix = create_feature_matrix(df_processed)
    if feature_matrix is not None:
        print("Feature matrix created successfully")

        # Call the calculate_similarity function
        similarity_matrix = calculate_similarity(feature_matrix)
        if similarity_matrix is not None:
            print("Similarity matrix created successfully")
        else:
            print("Failed to create similarity matrix")
    else:
        print("Failed to create feature matrix")
else:
    print("Preprocessing failed, cannot create feature matrix")

def get_recommendations_interface(title):
    # A function for the interface that calls get_recommendation() and returns a list of recommendations
    try:
        recommendations = get_recommendation(title, df, similarity_matrix)
        return "\n".join(recommendations)
    except Exception as e:
        return f"Error in receiving recommendations: {str(e)}"

# Define the Interface
app = gr.Interface(
    fn=get_recommendations_interface,
    inputs=gr.Textbox(label="Type your film/show here ..."),
    outputs=gr.Textbox(label="Recommendation"),
    title="AI Netflix Recommender",
    description="Ask for Netflix recommendations"
)

app.launch()