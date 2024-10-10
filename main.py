import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from docutils.parsers.rst.directives.misc import Class
from wordcloud import WordCloud
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from deep_translator import GoogleTranslator

# Download necessary resources for NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

class DataLoader:

    def __init__(self, filename, column_names, target):

        self.filename = filename
        self.target = target
        self.data = None
        self.labels = None
        self.column_names = column_names

        self._load_data(target)

    def _load_data(self, target):

        try:
            # Load the dataset
            self.data = pd.read_csv(self.filename, sep='\t', names=self.column_names)

            # Validate if the target column exists in the dataset
            if target not in self.data.columns:
                raise ValueError(f"Target column '{target}' not found in the dataset.")

            self.labels = self.data[target]

            print("Data loaded successfully.")

        except FileNotFoundError:
            print("File not found. Please check the file path.")

class DataManipulator(DataLoader):

    def __init__(self, filename, column_names, target):

        try:
            super().__init__(filename, column_names, target)
            print("\nData Description:")
            self._describe_variables()
        except FileNotFoundError:
            print("File not found. Please check the file path.")

    def _describe_variables(self):

        print("\nInformation of Data:")
        print(self.data.info())

        print("\nUnique values of features:")
        print(self.data.nunique())

        print("\nStatistical distribution of each variable:")
        print(self.data.describe())

class DataPreProcessing:

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.stop_words = set(stopwords.words('english'))

        # Sanity check
        self._sanity_check()

        # Apply text cleaning
        self._clean_text()

    def _sanity_check(self):
        try:
            if not self.data_loader:
                raise ValueError("DataLoader object is not provided.")
            if not isinstance(self.data_loader.data, pd.DataFrame):
                raise ValueError("Invalid DataLoader object. It should contain a pandas DataFrame.")
        except Exception as error:
            print(f"Error occurred: {error}")
            return False

    def _clean_text(self):
        # Lowercase the text
        self.data_loader.data['plot'] = self.data_loader.data['plot'].str.lower()

        # Remove punctuation
        self.data_loader.data['plot'] = self.data_loader.data['plot'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

        # Remove stopwords
        self.data_loader.data['plot'] = self.data_loader.data['plot'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in self.stop_words])
        )

class DataVisualization:

    def __init__(self, data_loader, valid_plot_types):

        self.data_loader = data_loader
        self.valid_plot_types = valid_plot_types
        self.labels = self.data_loader.data[self.data_loader.target].unique().tolist()

    def plots(self, plot_types):

        for plot_type in plot_types:
            # Check if the selected plots are in the list of available plots
            if plot_type not in self.valid_plot_types:
                print(
                    f"Ignoring invalid plot type: {plot_type}. Supported plot types: {', '.join(self.valid_plot_types)}")
                continue

            for feature in self.data_loader.data.columns:

                if plot_type == 'bar':
                    if feature == 'language':
                        # Plot the distribution of languages
                        self.data_loader.data[feature].value_counts().plot(kind='bar', figsize=(10, 6))
                        plt.title('Distribution of Languages')
                        plt.xlabel('Language')
                        plt.ylabel('Number of Movies')
                        plt.xticks(rotation=45)
                        plt.show()
                    if feature == 'director':
                        # Plot the top 10 directors by number of movies
                        self.data_loader.data[feature].value_counts().nlargest(10).plot(kind='bar', figsize=(10, 9))
                        plt.title('Top 10 Directors by Number of Movies')
                        plt.xlabel('Director')
                        plt.ylabel('Number of Movies')
                        plt.xticks(rotation=45)
                        plt.show()
                    if feature == 'genre':
                        # Create a crosstab of directors and genres (counts of movies per genre per director)
                        director_genre_crosstab = pd.crosstab(self.data_loader.data['director'], self.data_loader.data['genre'])
                        # Sum the number of movies per director (across all genres)
                        director_movie_counts = director_genre_crosstab.sum(axis=1)
                        # Get the top 10 directors with the most movies
                        top_10_directors = director_movie_counts.nlargest(10)
                        # Filter the original crosstab to show only the top 10 directors
                        top_directors_crosstab = director_genre_crosstab.loc[top_10_directors.index]
                        # Plot a stacked bar plot for the top 10 directors by genre
                        top_directors_crosstab.plot(kind='bar', stacked=True, figsize=(10, 9))
                        plt.title('Top 10 Directors by Genre')
                        plt.xlabel('Director')
                        plt.ylabel('Number of Movies')
                        plt.xticks(rotation=45)
                        plt.show()

                if plot_type == 'hist' and feature == 'plot':
                    # Calculate the length of each plot
                    self.data_loader.data['plot_length'] = self.data_loader.data['plot'].apply(lambda x: len(str(x).split()))
                    # Plot the histogram
                    self.data_loader.data['plot_length'].plot(kind='hist', bins=50, figsize=(10, 6))
                    plt.title('Distribution of Movie Plot Lengths')
                    plt.xlabel('Number of Words in Plot')
                    plt.ylabel('Frequency')
                    plt.show()

                if plot_type == 'pie' and feature == 'genre':
                    # Plot the distribution of genres
                    self.data_loader.data[feature].value_counts().plot(kind='pie', figsize=(10, 6))
                    plt.title(f'Distribution of Genres')
                    plt.show()

                if plot_type == 'wordcloud' and feature == 'plot':
                    # Combine all the plots into a single string
                    plot_text = ' '.join(self.data_loader.data['plot'].dropna())
                    # Create the word cloud
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(plot_text)
                    # Display the word cloud
                    plt.figure(figsize=(10, 6))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.show()

class FeatureCreation:

    def __init__(self, data_loader):

        self.data_loader = data_loader

    def create_text_based_features(self):

        # Create Plot Length Feature
        self.data_loader.data['plot_length'] = self.data_loader.data['plot'].apply(lambda x: len(str(x).split()))
        print("Created plot_length feature\n")
        # Create Average Word Length Feature
        self.data_loader.data['avg_word_length'] = self.data_loader.data['plot'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]))
        print("Created avg_word_length feature\n")
        # Create Unique Word Count Feature
        self.data_loader.data['unique_word_count'] = self.data_loader.data['plot'].apply(
            lambda x: len(set(str(x).split())))
        print("Created unique_word_count feature\n")
        # Create Sentiment Polarity Feature
        self.data_loader.data['sentiment_polarity'] = self.data_loader.data['plot'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity)
        print("Created sentiment_polarity feature\n")

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return set(synonyms)


def synonym_replacement(text, n_replacements):
    words = word_tokenize(text)
    new_words = words.copy()

    # List of indices where replacements will happen
    indices_to_replace = random.sample(range(len(words)), min(n_replacements, len(words)))
    successful_replacements = 0
    replacements_log = []

    for idx in indices_to_replace:
        word = words[idx]
        synonyms = get_synonyms(word)
        if synonyms:
            # Replace the word with a random synonym if available
            synonym = random.choice(list(synonyms))

            new_words[idx] = synonym
            successful_replacements += 1
            replacements_log.append((word, synonym))
    new_plot = ' '.join(new_words)
    # print(f"Original plot: {text}...\nNew plot: {new_plot}...\n")
    # print(f"Replaced {successful_replacements} words in the plot.")
    # for original, new in replacements_log:
    #     print(f"Replaced '{original}' with '{new}'")
    return new_plot


def augment_data(X_train, n_augmentations):
    new_rows = []

    # Iterate over each row in the dataset
    for _, row in X_train.iterrows():
        original_row = row.to_dict()  # Convert the row to a dictionary

        # Generate n_augmentations new rows with modified plots
        for _ in range(n_augmentations):
            new_row = original_row.copy()  # Keep other columns the same
            new_row['plot'] = synonym_replacement(row['plot'], n_replacements=50)
            new_rows.append(new_row)

    # Create a DataFrame from the new rows and append to X_train
    augmented_df = pd.DataFrame(new_rows)
    return pd.concat([X_train, augmented_df], ignore_index=True)

def back_translate(text, src_lang='en', mid_lang='es'):
    # Translate text to the intermediate language (e.g., Spanish)
    translated = GoogleTranslator(source=src_lang, target=mid_lang).translate(text)
    # Translate it back to the original language
    back_translated = GoogleTranslator(source=mid_lang, target=src_lang).translate(translated)
    return back_translated

# %% 1- Pre Processing and EDA

filename = 'data/train.txt'
column_names = ['title', 'language', 'genre', 'director', 'plot']

# Load the data
data_loader = DataManipulator(filename, column_names,'genre')

# Preprocess the data
data_preprocessing = DataPreProcessing(data_loader)

data_loader.data.to_csv('data/movie_data_preprocessed.csv', index=False)

# Visualize the data
data_visualization = DataVisualization(data_loader, ['pie', 'bar', 'hist', 'wordcloud'])
data_visualization.plots(['pie', 'bar', 'hist', 'wordcloud'])

# %% 2- Feature Creation

# Initialize the FeatureCreation class with the data
feature_creator = FeatureCreation(data_loader)

print("\nFeatures Created:\n")

# Create text-based features
feature_creator.create_text_based_features()

data_loader.data.to_csv('data/movie_data_featurecreation.csv', index=False)

# Data Visualization of the new features

# %% 3- Data Splits

# Divide the data into features and target variable
X = data_loader.data.drop(columns=[data_loader.target])
y = data_loader.data[data_loader.target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample
X_train_augmented = augment_data(X_train, n_augmentations=1)
original_size = len(X_train)
print(f"Original dataset size: {original_size}")
print(f"Augmented dataset size: {len(X_train_augmented)}")
#save the augmented data
# feature_creator_augmented = FeatureCreation(X_train_augmented)
# feature_creator_augmented.create_text_based_features()
# text = "A thrilling adventure of a young boy in the mysterious woods."
# back_translated_text = back_translate(text)
# print(back_translated_text)
#X_train_augmented.to_csv('data/movie_data_augmented.csv', index=False)


# Split the training data into training and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)