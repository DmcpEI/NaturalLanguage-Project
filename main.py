import pandas as pd

# Assuming your tab-separated file is named 'train.txt'
file_path = 'train.txt'

# Define the column names based on the structure
column_names = ['title', 'language', 'genre', 'director', 'plot']

# Read the tab-separated file into a DataFrame
data = pd.read_csv(file_path, sep='\t', names=column_names)

# Save the DataFrame to a CSV file
csv_output_path = 'movie_data.csv'
data.to_csv(csv_output_path, index=False)

# Get the unique genres/languages
unique_genres = data['genre'].unique()
unique_languages = data['language'].unique()
print(f"Unique genres: {unique_genres}")
print(f"Unique languages: {unique_languages}")

# Count how many movies fall under each genre/language
genre_counts = data['genre'].value_counts()
language_counts = data['language'].value_counts()
print("\nMovies per genre:")
print(genre_counts)
print("\nMovies per language:")
print(language_counts)
