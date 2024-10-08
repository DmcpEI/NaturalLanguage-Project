import pandas as pd
import re
import nltk
import random
from docutils.parsers.rst.directives.misc import Class
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download necessary resources for NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

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
        self.data_loader.data['clean_plot'] = self.data_loader.data['plot'].str.lower()

        # Check if there are any NaN values in the 'clean_plot' column
        #nan_count = self.data_loader.data['clean_plot'].isna().sum()
        #print(f"Number of NaN values in 'clean_plot': {nan_count}")

        # Remove punctuation
        #self.data_loader.data['clean_plot'] = self.data_loader.data['clean_plot'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

        # Remove stopwords
        #self.data_loader.data['clean_plot'] = self.data_loader.data['clean_plot'].apply(
        #    lambda x: ' '.join([word for word in x.split() if word not in self.stop_words]))

        # Debug: Check the number of rows after cleaning
        print(f"Number of rows after cleaning: {self.data_loader.data.shape[0]}")

    # Method to save data to CSV
    def save_to_csv(self, output_path):
        try:
            self.data_loader.data.to_csv(output_path, index=False)
            print(f"Data successfully saved to {output_path}")
        except Exception as e:
            print(f"Error saving CSV: {e}")



def synonym_replacement(text, num_replacements):
    words = word_tokenize(text)
    new_samples = []

    for _ in range(1000):  # Generate 10 different variations
        new_words = words.copy()
        random_indices = random.sample(range(len(words)), min(num_replacements, len(words)))  # Select random indices to replace
        for idx in random_indices:
            synonyms = wordnet.synsets(words[idx])
            if synonyms:  # If the word has synonyms
                synonym = random.choice(synonyms).lemmas()[0].name()  # Pick a random synonym
                new_words[idx] = synonym.replace('_', ' ')  # Replace with the synonym
        new_samples.append(' '.join(new_words))  # Add the new variation

    return new_samples


# Example usage
# original_sentence = "The quick brown fox jumps over the lazy dog."
# augmented_sentence = synonym_replacement(original_sentence, n=2)
# print("Original:", original_sentence)
# print("Augmented:", augmented_sentence)


# %% 1- Pre Processing and EDA

filename = 'data/train.txt'
column_names = ['title', 'language', 'genre', 'director', 'plot']

# Load and preprocess the data
data_loader = DataManipulator(filename, column_names ,'genre')
data_preprocessing = DataPreProcessing(data_loader)

print(data_loader.data.shape)

data_preprocessing.save_to_csv('data/movie_data_preprocessed.csv')

# # Data Cleaning
#
# # Data Visualization
#
# Divide the data into features and target variable
X = data_loader.data.drop(columns=[data_loader.target])
y = data_loader.data[data_loader.target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample
original_size = len(X_train)
print(f"Original dataset size: {original_size}")

augmented_texts = []

for text in X_train:
    augmented_texts.extend(synonym_replacement(text, num_replacements=1))  # Generate multiple samples per text

# Create a DataFrame for the augmented texts
df_augmented = pd.DataFrame({'text': augmented_texts})

# Combine original and augmented data
df_combined = pd.concat([X_train.reset_index(drop=True), df_augmented.reset_index(drop=True)], ignore_index=True)

# Remove duplicates
#df_combined = df_combined.drop_duplicates().reset_index(drop=True)

# Display sizes
new_size = len(df_combined)
print(f"New dataset size after augmentation: {new_size}")

# Split the training data into training and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)