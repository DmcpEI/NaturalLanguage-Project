import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary resources for NLTK
nltk.download('stopwords')
nltk.download('wordnet')

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

        self._clean_text()

        self._vectorize_text()

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

        # Remove punctuation
        self.data_loader.data['clean_plot'] = self.data_loader.data['clean_plot'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

        # Remove stopwords
        self.data_loader.data['clean_plot'] = self.data_loader.data['clean_plot'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in self.stop_words]))

    def _vectorize_text(self):
        # Vectorize the cleaned text using TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.X_tfidf = vectorizer.fit_transform(self.data_loader.data['clean_plot'])

        # Optionally, you can save the vectorizer to use later for transforming new data
        self.vectorizer = vectorizer


# %% 1- Pre Processing and EDA

filename = 'data/train.txt'
column_names = ['title', 'language', 'genre', 'director', 'plot']

data_loader = DataManipulator(filename, column_names ,'genre')

data_preprocessing = DataPreProcessing(data_loader)

data_loader.data.to_csv('data/movie_data_preprocessed.csv', index=False)
