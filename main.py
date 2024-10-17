import joblib
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import os
import pandas as pd
import pickle
from pycm import ConfusionMatrix
import random
import re
import spacy
import textstat

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.base import clone
from textblob import TextBlob
import warnings

from transformers import BertTokenizer, BertModel
from wordcloud import WordCloud

# Download necessary resources for NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

nlp = spacy.load('en_core_web_sm')

warnings.filterwarnings("ignore")

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


class DataProcessing:

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.stop_words = set(stopwords.words('english'))

        self._sanity_check()

    def _sanity_check(self):
        try:
            if not self.data_loader:
                raise ValueError("DataLoader object is not provided.")
            if not isinstance(self.data_loader.data, pd.DataFrame):
                raise ValueError("Invalid DataLoader object. It should contain a pandas DataFrame.")
        except Exception as error:
            print(f"Error occurred: {error}")
            return False

    def clean_text(self, data = None):

        if data is None:
            # Lowercase the text
            data_loader.data['plot'] = data_loader.data['plot'].str.lower()
            # Remove punctuation
            data_loader.data['plot'] = data_loader.data['plot'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
            # Remove stopwords
            data_loader.data['plot'] = data_loader.data['plot'].apply(
                lambda x: ' '.join([word for word in x.split() if word not in self.stop_words]))
        else:
            data['plot'] = data['plot'].str.lower()
            data['plot'] = data['plot'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
            data['plot'] = data['plot'].apply(
                lambda x: ' '.join([word for word in x.split() if word not in self.stop_words]))
            return data


class BERTEmbeddings:
    #load pre-trained BERT model and tokenizer
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_embeddings(self, text):
        #tokenize the text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

        #get the embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            plot_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return plot_embeddings


class Embeddings:

    def __init__(self, glove_file_path=None):

        self.bert_embedder = BERTEmbeddings()

        # Load GloVe embeddings if the file path is provided
        if glove_file_path:
            self.embeddings_index = self._load_glove_embeddings(glove_file_path)
        else:
            self.embeddings_index = None

    def _load_glove_embeddings(self, file_path):
        """Load GloVe pre-trained embeddings into a dictionary."""
        embeddings_index = {}
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print(f"Loaded {len(embeddings_index)} word vectors from GloVe.")
        return embeddings_index

    def _glove_plot_to_embedding(self, plot):
        """Convert a plot of text into a dense vector representation using GloVe."""
        words = plot.split()
        embedding_dim = 100  # Adjust based on GloVe vector size
        plot_vector = np.zeros((embedding_dim,))
        word_count = 0

        for word in words:
            word_vector = self.embeddings_index.get(word.lower())
            if word_vector is not None:
                plot_vector += word_vector
                word_count += 1

        if word_count > 0:
            plot_vector /= word_count
        return plot_vector

    def _bert_plot_to_embedding(self, plot):
        """Convert a plot of text into a dense vector representation using BERT."""
        embedding = self.bert_embedder.get_embeddings(plot)  # Call the BERT embedding method
        return embedding

    def _plot_to_embedding(self, plot, method="glove"):
        """
        Convert a plot to an embedding using the selected method (GloVe or BERT).

        Parameters:
        plot (str): The plot text to embed.
        method (str): The embedding method to use ("glove" or "bert").

        Returns:
        np.array: The resulting embedding vector.
        """
        if method == "glove":
            return self._glove_plot_to_embedding(plot)
        elif method == "bert":
            return self._bert_plot_to_embedding(plot)
        else:
            raise ValueError(f"Unknown embedding method: {method}")

    def embedding_df(self, X, method="glove"):
        """Create plot embeddings for X using the selected embedding method."""
        X['plot_embedding'] = X['plot'].apply(lambda plot: self._plot_to_embedding(plot, method=method))

        # Stack the plot embeddings into a matrix
        plot_embeddings = np.vstack(X['plot_embedding'].values)

        plot_embeddings_df = pd.DataFrame(plot_embeddings,
                                          columns=[f'embedding_{i}' for i in range(plot_embeddings.shape[1])])

        return plot_embeddings_df

    def embedding_X(self, X_train_raw, X_test_raw, method="glove"):
        """
        Create dense vector representations for both train and test data and return encoded X and y.

        Parameters:
        method (str): The embedding method to use ("glove" or "bert").
        """

        # ----- Processing X_train_augmented -----
        plot_embeddings_train_df = self.embedding_df(X_train_raw, method=method)

        X_train_raw.reset_index(drop=True, inplace=True)
        plot_embeddings_train_df.reset_index(drop=True, inplace=True)

        X_train_processed = pd.concat([X_train_raw.drop(columns=['plot', 'title', 'plot_embedding']),
                                       plot_embeddings_train_df], axis=1)

        # Initialize label encoders dictionary
        label_encoders_X = {}

        # Encode 'language' and 'director' in training set
        for col in ['language', 'director']:
            le = LabelEncoder()
            X_train_processed[col] = le.fit_transform(X_train_processed[col].astype(str))
            label_encoders_X[col] = le  # Save the encoder to use for test data later

        # ----- Processing X_test_raw -----
        plot_embeddings_test_df = self.embedding_df(X_test_raw, method=method)

        X_test_raw.reset_index(drop=True, inplace=True)
        plot_embeddings_test_df.reset_index(drop=True, inplace=True)

        X_test_processed = pd.concat([X_test_raw.drop(columns=['plot', 'title', 'plot_embedding']),
                                      plot_embeddings_test_df], axis=1)

        # Encode 'language' and 'director' in test set, handling unseen labels
        for col in ['language', 'director']:
            X_test_processed[col] = X_test_processed[col].apply(
                lambda x: label_encoders_X[col].transform([x])[0] if x in label_encoders_X[col].classes_ else -1
            )

        # ----- Standardizing feature matrices -----
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_processed)
        X_test_scaled = scaler.transform(X_test_processed)

        print(
            f"{method.capitalize()} embeddings created and concatenated with other features for both training and test sets.")
        return X_train_scaled, X_test_scaled, scaler, label_encoders_X

    def encode_target(self, y_train_raw, y_test_raw, ):

        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train_raw)
        y_test_encoded = le.fit_transform(y_test_raw)

        return y_train_encoded, y_test_encoded

    def embedding_test(self, test_data, scaler, label_encoders_X, method="glove"):

        plot_embeddings_test_data_df = self.embedding_df(test_data, method=method)

        test_data.reset_index(drop=True, inplace=True)
        plot_embeddings_test_data_df.reset_index(drop=True, inplace=True)

        test_data_processed = pd.concat([test_data.drop(columns=['plot', 'title', 'plot_embedding']),
                                       plot_embeddings_test_data_df], axis=1)

        for col in ['language', 'director']:
            test_data_processed[col] = test_data_processed[col].apply(
                lambda x: label_encoders_X[col].transform([x])[0] if x in label_encoders_X[col].classes_ else -1
            )

        test_data_scaled = scaler.transform(test_data_processed)

        print(f"{method.capitalize()} embeddings created and concatenated with other features for the test set with no labels.")
        return test_data_scaled


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
    def __init__(self):
        pass

    def create_features(self, data):
        # Existing features
        data['plot_length'] = data['plot'].apply(lambda x: len(str(x).split()))
        print("Created plot_length feature\n")

        data['sentiment_polarity'] = data['plot'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        data['sentiment_subjectivity'] = data['plot'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
        print("Created sentiment_polarity and sentiment_subjectivity features\n")

        # Lexical Diversity (Type-Token Ratio)
        data['type_token_ratio'] = data['plot'].apply(lambda x: len(set(str(x).split())) / max(1, len(str(x).split())))
        print("Created type_token_ratio feature\n")

        # Average Word Length
        data['avg_word_length'] = data['plot'].apply(self._average_word_length)
        print("Created avg_word_length feature\n")

        # Unique Word Count
        data['unique_word_count'] = data['plot'].apply(self._unique_word_count)
        print("Created unique_word_count feature\n")

        # POS Tags Ratios
        data['pos_noun_ratio'], data['pos_verb_ratio'], data['pos_adj_ratio'] = zip(*data['plot'].apply(self._pos_tag_ratios))
        print("Created POS tag ratio features\n")

        # Named Entity Count
        data['entity_count'] = data['plot'].apply(self._entity_count)
        print("Created named entity count feature\n")

        # Flesch-Kincaid Readability Score
        data['flesch_kincaid'] = data['plot'].apply(self._flesch_kincaid_score)
        print("Created Flesch-Kincaid readability score feature\n")

        # Topic Modeling with LDA
        data['lda_topic_0'], data['lda_topic_1'] = self._lda_topic_features(data['plot'])
        print("Created LDA topic features\n")

    def _average_word_length(self, text):
        """Calculate average word length in the plot."""
        words = str(text).split()
        if len(words) == 0:
            return 0
        return sum(len(word) for word in words) / len(words)

    def _unique_word_count(self, text):
        """Count the number of unique words in the plot."""
        words = str(text).split()
        return len(set(words))

    def _pos_tag_ratios(self, text):
        """Calculate the ratio of nouns, verbs, and adjectives in the plot."""
        doc = nlp(text)
        total_tokens = len(doc)
        noun_count = sum(1 for token in doc if token.pos_ == 'NOUN')
        verb_count = sum(1 for token in doc if token.pos_ == 'VERB')
        adj_count = sum(1 for token in doc if token.pos_ == 'ADJ')

        return noun_count / total_tokens, verb_count / total_tokens, adj_count / total_tokens

    def _entity_count(self, text):
        """Count the number of named entities in the plot."""
        doc = nlp(text)
        return len(doc.ents)

    def _flesch_kincaid_score(self, text):
        """Calculate the Flesch-Kincaid readability score."""
        return textstat.flesch_kincaid_grade(text)

    def _lda_topic_features(self, plots):
        """Generate LDA topic features."""
        vectorizer = TfidfVectorizer(max_features=500)
        X = vectorizer.fit_transform(plots)

        lda = LatentDirichletAllocation(n_components=2, random_state=42)
        lda_topics = lda.fit_transform(X)

        return lda_topics[:, 0], lda_topics[:, 1]


class FeatureEvaluator:
    def __init__(self, X_train, y_train, model=None):
        """
        Initialize the FeatureEvaluator with training data and a model.

        Parameters:
        X_train (DataFrame or numpy array): The training features.
        y_train (DataFrame or numpy array): The target variable.
        model (sklearn model): A machine learning model. Default is RandomForestClassifier.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.model = model if model is not None else RandomForestClassifier(n_estimators=100, random_state=42)

        # Check if X_train is a DataFrame to extract feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns
        else:
            self.feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

    def evaluate_importance(self):
        """
        Train the model and calculate feature importance.
        """
        # Fit the model
        self.model.fit(self.X_train, self.y_train)

        # Get feature importances
        self.importances = self.model.feature_importances_
        self.indices = np.argsort(self.importances)[::-1]  # Sort in descending order

        # Get feature names
        if isinstance(self.X_train, pd.DataFrame):
            feature_names = self.X_train.columns
        else:
            feature_names = [f"Feature {i}" for i in range(self.X_train.shape[1])]

        # Print the ranking of features with names
        print("Feature ranking:")
        for i in range(self.X_train.shape[1]):
            print(f"{i + 1}. {feature_names[self.indices[i]]} ({self.importances[self.indices[i]]})")

        return self.importances

    def plot_importance(self):
        """
        Plot the feature importance.
        """
        # Plot the feature importances
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(self.X_train.shape[1]), self.importances[self.indices], align="center")
        plt.xticks(range(self.X_train.shape[1]), [self.feature_names[i] for i in self.indices], rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.xlim([-1, self.X_train.shape[1]])
        plt.tight_layout()
        plt.show()

    def get_top_features(self, n=10):
        """
        Get the top N features based on importance.

        Parameters:
        n (int): Number of top features to return.

        Returns:
        list: Names of the top N features.
        """
        top_indices = self.indices[:n]
        top_features = [self.feature_names[i] for i in top_indices]
        return top_features


class DataAugmentation:

    def __init__(self):
        pass

    def random_insertion(self, text, n_insertions=3):
        """Randomly insert adjectives or relevant genre words into the plot."""
        words = word_tokenize(text)
        insertions = ["dark", "mysterious", "frightening", "intense", "haunting", "thrilling"]

        for _ in range(n_insertions):
            insert_word = random.choice(insertions)
            insert_position = random.randint(0, len(words))
            words.insert(insert_position, insert_word)

        new_text = ' '.join(words)
        return new_text

    def random_deletion(self, text, p=0.2):
        """Randomly delete words from the plot with probability p."""
        words = word_tokenize(text)
        new_words = [word for word in words if random.uniform(0, 1) > p]

        new_text = ' '.join(new_words)
        return new_text

    def augment_data(self, data, n_augmentations=2):
        original_row_count = len(data)
        new_rows = []

        # Iterate over each row in the dataset
        for _, row in data.iterrows():
            original_row = row.to_dict()  # Convert the row to a dictionary

            # Generate n_augmentations new rows with modified plots
            for _ in range(n_augmentations):
                new_row = original_row.copy()  # Keep other columns the same

                # Apply random insertion or random deletion
                if random.random() > 0.5:
                    new_row['plot'] = self.random_insertion(row['plot'])
                else:
                    new_row['plot'] = self.random_deletion(row['plot'])

                new_rows.append(new_row)

        # Create a DataFrame from the new rows and append to the original data
        augmented_df = pd.DataFrame(new_rows)
        augmented_data = pd.concat([data, augmented_df], ignore_index=True)

        # Insights into the augmentation process
        augmented_row_count = len(augmented_df)
        total_row_count = len(augmented_data)
        print(f"Original rows: {original_row_count}")
        print(f"Augmented rows: {augmented_row_count}")
        print(f"Total rows after augmentation: {total_row_count}")

        return augmented_data


class ModelOptimization:

    def __init__(self, X_train, y_train, X_val, y_val):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def optimize_logistic_regression(self, C_values=(0.01, 0.1, 1.0, 10.0), penalty=(None, 'l2')):

        best_accuracy = -1
        best_c = None
        best_penalty = None

        for c in C_values:
            for penalty_selected in penalty:
                lr = LogisticRegression(C=c, penalty=penalty_selected, solver='saga', multi_class='auto', max_iter=100000)
                lr.fit(self.X_train, self.y_train)  # Fit the model to the training data
                accuracy = lr.score(self.X_val, self.y_val)  # Calculate the accuracy on the validation data

                print(f"C = {c}, Penalty = {penalty_selected}, Accuracy = {accuracy}")

                if accuracy > best_accuracy:  # Update the best parameters if the accuracy is higher
                    best_accuracy = accuracy
                    best_c = c
                    best_penalty = penalty_selected

        print("Best C value:", best_c)
        print("Best penalty:", best_penalty)
        print("Best accuracy:", best_accuracy)
        return best_c, best_penalty

    def optimize_random_forest(self, n_estimators=(100, 200), max_depth=(None, 10, 20), min_samples_split=(2, 5)):
        """
        Optimizes the parameters for Random Forest classifier.

        Parameters:
            n_estimators (tuple): Values to try for number of trees. Default is (100, 200).
            max_depth (tuple): Values to try for maximum depth. Default is (None, 10, 20).
            min_samples_split (tuple): Values to try for minimum samples required to split an internal node. Default is (2, 5).

        Returns:
            dict: Best parameters for Random Forest (n_estimators, max_depth, min_samples_split).
        """
        best_accuracy = -1
        best_params = None

        for n in n_estimators:
            for depth in max_depth:
                for min_samples in min_samples_split:
                    rf = RandomForestClassifier(n_estimators=n, max_depth=depth, min_samples_split=min_samples,
                                                random_state=42)
                    rf.fit(self.X_train, self.y_train)  # Fit the model to the training data
                    accuracy = rf.score(self.X_val, self.y_val)  # Calculate the accuracy on the validation data

                    print(
                        f"n_estimators = {n}, max_depth = {depth}, min_samples_split = {min_samples}, Accuracy = {accuracy}")

                    if accuracy > best_accuracy:  # Update the best parameters if the accuracy is higher
                        best_accuracy = accuracy
                        best_params = (n, depth, min_samples)

        print("Best Random Forest parameters:", best_params)
        print("Best accuracy:", best_accuracy)
        return best_params

    def optimize_svm(self, C_values=(0.1, 1, 10), kernel=('linear', 'rbf'), gamma=('scale', 'auto')):
        """
        Optimizes the parameters for SVM classifier.

        Parameters:
            C_values (tuple): Values to try for regularization parameter C. Default is (0.1, 1, 10).
            kernel (tuple): Kernel types to try. Default is ('linear', 'rbf').
            gamma (tuple): Gamma values to try for the RBF kernel. Default is ('scale', 'auto').

        Returns:
            tuple: Best parameters for SVM (C, kernel, gamma).
        """
        best_accuracy = -1
        best_params = None

        for c in C_values:
            for k in kernel:
                for g in gamma:
                    svm = SVC(C=c, kernel=k, gamma=g)
                    svm.fit(self.X_train, self.y_train)  # Fit the model to the training data
                    accuracy = svm.score(self.X_val, self.y_val)  # Calculate the accuracy on the validation data

                    print(f"C = {c}, kernel = {k}, gamma = {g}, Accuracy = {accuracy}")

                    if accuracy > best_accuracy:  # Update the best parameters if the accuracy is higher
                        best_accuracy = accuracy
                        best_params = (c, k, g)

        print("Best SVM parameters:", best_params)
        print("Best accuracy:", best_accuracy)
        return best_params

    def optimize_mlp(self, population_size=20, max_generations=50, layer_range=(1, 100),
                     activation=('logistic', 'tanh')):
        """
        Optimizes the parameters for Multi-layer Perceptron (MLP) classifier.

        Parameters:
            population_size (int): Size of the population for optimization. Default is 20.
            max_generations (int): Maximum number of generations for optimization. Default is 50.
            layer_range (tuple): Range of neurons in hidden layers. Default is (1, 100).
            activation (tuple): Activation functions to try. Default is ('logistic', 'tanh').

        Returns:
            tuple: Best parameters for MLP (number of neurons, activation function), best model, and best accuracy.
        """
        # Initialize the population
        population = []
        for _ in range(population_size):
            neurons = random.randint(*layer_range)
            activation_selected = random.choice(activation)
            population.append((neurons, activation_selected))

        best_model = None  # To store the best model
        best_params = None  # To store the best parameters
        best_accuracy = 0  # To store the best accuracy

        # Random evolutionary search
        for generation in range(max_generations):
            print(f"Generation {generation + 1}/{max_generations}")
            new_population = []
            for i, (neurons, activation_selected) in enumerate(population):
                # Skip the first iteration from generation 1 onwards since it is the best element found in the previous iteration
                if generation != 0 and i == 0:
                    new_population.append((best_params, best_accuracy))
                    continue
                print("Generation ", generation, " element ", i)

                # Train the MLP model
                mlp = MLPClassifier(hidden_layer_sizes=(neurons,), activation=activation_selected, early_stopping=True)
                mlp.fit(self.X_train, self.y_train)
                accuracy = mlp.score(self.X_val, self.y_val)
                new_population.append(((neurons, activation_selected), accuracy))
                print(f"Neurons: {neurons}, activation: {activation_selected}, Accuracy: {accuracy}")

                # Update the best model, params, and accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = (neurons, activation_selected)
                    best_model = mlp

            new_population.sort(key=lambda x: x[1], reverse=True)
            best_params, best_accuracy = new_population[0]
            population = [best_params]

            # Use the parameters of the best individual to bias the generation of new individuals
            best_neurons, best_activation = best_params

            # Break when only 2 individuals are left
            if population_size <= 2:
                break
            population_size -= 1

            # Generate the new population
            for _ in range(1, population_size):
                # Randomly generate neurons with a bias towards the best_neurons
                neurons = random.randint(int(np.ceil(best_neurons - 20)) + 1, int(best_neurons + 20))
                # Randomly select activation function
                activation_selected = random.choice(activation)
                population.append((neurons, activation_selected))

        print("Best MLP parameters:", best_params)
        print("Best accuracy:", best_accuracy)
        return best_params, best_model


class CrossValidator:

    def __init__(self, k=5):

        self.k = k
        self.kf = KFold(n_splits=k, shuffle=True)
        self.cm = None
        self.accuracy_scores = []
        self.sensitivity_scores = []
        self.specificity_scores = []
        self.f1_scores = []  # Add a list to store F1-scores
    def cross_validate(self, model, X, y):

        for train_index, val_index in self.kf.split(X):  # Split the data into training and validation sets
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model.fit(X_train, y_train)  # Fit the model to the training data
            y_pred = model.predict(X_val)  # Predict the labels for the validation data

            self.cm = ConfusionMatrix(actual_vector=list(y_val), predict_vector=list(y_pred))

            self.accuracy_scores.append(accuracy_score(y_val, y_pred))
            self.sensitivity_scores.append(float(self.cm.TPR_Macro))
            self.specificity_scores.append(float(self.cm.TNR_Macro))
            self.f1_scores.append(f1_score(y_val, y_pred, average='macro'))  # Calculate F1-score

        avg_accuracy = sum(self.accuracy_scores) / len(self.accuracy_scores)
        avg_sensitivity = sum(self.sensitivity_scores) / len(self.sensitivity_scores)
        avg_specificity = sum(self.specificity_scores) / len(self.specificity_scores)
        avg_f1_score = sum(self.f1_scores) / len(self.f1_scores)  # Average F1-score

        return avg_accuracy, avg_sensitivity, avg_specificity, avg_f1_score

    def evaluate_on_test_set(self, model, X_test, y_test):

        y_pred = model.predict(X_test)
        cm = ConfusionMatrix(actual_vector=list(y_test), predict_vector=list(y_pred))
        accuracy = cm.Overall_ACC
        sensitivity = cm.TPR_Macro
        specificity = cm.TNR_Macro
        f1 = f1_score(y_test, y_pred, average='macro')  # F1-score on the test set
        return accuracy, sensitivity, specificity, f1


class ModelBuilding:

    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val, encoder, k=5, save_all=True):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.encoder = encoder
        self.k = k
        self.save_all = save_all
        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.best_score = -1
        #self.best_test_score = -1
        self.best_model_changed = False
        self.history = {}

    def build_models(self, model_name, models_dict, results_dict):
        """
        Builds, optimizes and evaluates machine learning models.

        Parameters:
            model_name (str): Name of the model to build.
            models_dict (dict): Dictionary containing the models to build.
            results_dict (dict): Dictionary to store the results of the models.

        Raises:
            ValueError: If the model type is not supported.
        """

        model_optimization = ModelOptimization(self.X_train, self.y_train, self.X_val, self.y_val)
        print("\nTraining", model_name, "model")
        for name, model_params in models_dict.items():  # Iterate over the models
            if model_name != name:
                continue
            model = model_params.pop('model')
            model_params_check = {}

            if model_name == "LogisticRegression":  # Optimize the parameters for Logistic Regression
                lr_params = model_optimization.optimize_logistic_regression(**model_params)
                model_params_check['C'] = lr_params[0]
                model_params_check['penalty'] = lr_params[1]
                params = lr_params
            elif model_name == "RandomForest":  # Optimize the parameters for Random Forest
                rf_params = model_optimization.optimize_random_forest(**model_params)
                model_params_check['n_estimators'] = rf_params[0]
                model_params_check['max_depth'] = rf_params[1]
                model_params_check['min_samples_split'] = rf_params[2]
                params = rf_params
            elif model_name == "SVM":  # Optimize the parameters for SVM
                svm_params = model_optimization.optimize_svm(**model_params)
                model_params_check['C'] = svm_params[0]
                model_params_check['kernel'] = svm_params[1]
                model_params_check['gamma'] = svm_params[2]
                params = svm_params
            elif model_name == "MLP":  # Optimize the parameters for MLP
                mlp_params = model_optimization.optimize_mlp(**model_params)
                model_params_check['hidden_layer_sizes'] = (mlp_params[0],)
                model_params_check['activation'] = mlp_params[1]
                params, mlp_model = mlp_params
            else:
                raise ValueError("Model type is not supported.")

            if model_name == "MLP":
                model_instance = mlp_model
            else:
                model_instance = model(**model_params_check)  # Create an instance of the model with the optimized parameters
                model_instance.fit(self.X_train, self.y_train)  # Fit the model to the training data

            val_score = model_instance.score(self.X_val, self.y_val)  # Calculate the accuracy on the validation data
            self.history[str(name)] = val_score  # Store the validation score in the history dictionary

            if val_score > self.best_score:  # Update the best model if the validation score is higher
                print("\nNew best model found!")
                self.best_score = val_score
                self.best_model = model_instance
                self.best_model_name = name
                self.best_params = params
                self.best_model_checked = model
                self.best_model_params_checked = model_params_check
                self.best_model_changed = True  # Update flag when a new best model is found
            else:
                self.best_model_changed = False  # Reset flag if the best model didn't change

            if self.save_all:  # Save all models if the flag is set
                self.save_model(model_instance, name)

            self.kf_cv = CrossValidator(k=self.k)

            print(f"\nPreforming cross-validation on the {model_name} model:")

            # Performance of the model during cross-validation
            avg_accuracy_cv, avg_sensitivity_cv, avg_specificity_cv, avg_f1_score_cv = self.kf_cv.cross_validate(
                model_instance, self.X_train, self.y_train)

            print("Average accuracy during cross-validation:", avg_accuracy_cv)
            print("Average sensitivity during cross-validation:", avg_sensitivity_cv)
            print(f"Average specificity during cross-validation: {avg_specificity_cv}\n")
            print(f"Average F1-score during cross-validation: {avg_f1_score_cv}\n")

            print(f"\nPerformance of the {model_name} model on the Test set:")

            # Performance of the model on the test set
            accuracy_test, sensitivity_test, specificity_test, f1_score_test = self.kf_cv.evaluate_on_test_set(
                model_instance,
                self.X_test,
                self.y_test)
            print("Test set accuracy:", accuracy_test)
            print("Test set sensitivity:", sensitivity_test)
            print(f"Test set specificity: {specificity_test}\n")
            print(f"Test set F1-score: {f1_score_test}\n")

            # Store the results in the results dictionary
            if self.encoder == 'glove':
                results_dict[model_name + ' with Glove Embeddings']['Accuracy'] = float(accuracy_test)
                results_dict[model_name + ' with Glove Embeddings']['Sensitivity'] = float(sensitivity_test)
                results_dict[model_name + ' with Glove Embeddings']['Specificity'] = float(specificity_test)
                results_dict[model_name + ' with Glove Embeddings']['F1_score'] = float(f1_score_test)
            elif self.encoder == 'bert':
                results_dict[model_name + ' with BERT Embeddings']['Accuracy'] = float(accuracy_test)
                results_dict[model_name + ' with BERT Embeddings']['Sensitivity'] = float(sensitivity_test)
                results_dict[model_name + ' with BERT Embeddings']['Specificity'] = float(specificity_test)
                results_dict[model_name + ' with BERT Embeddings']['F1_score'] = float(f1_score_test)

            # Print the best model and its parameters
            print("\nOptimization finished, history:\n")
            print("Model name\t\tAccuracy")
            for model, accuracy in self.history.items():
                print(f"{model}\t\t{accuracy}")
            print("\nBest performing model:", self.best_model_name)
            print("Best validation score:", self.best_score)
            print("Best parameters:", self.best_params)

            if not self.save_all:  # Save the best model if the flag is not set
                self.save_model(self.best_model, self.best_model_name)

    def save_model(self, model, filename):
        """
        Saves the model to a file.

        Parameters:
            model: Trained machine learning model.
            filename (str): Name of the file to save the model.
        """

        folder_path = "./models"
        full_path = os.path.join(folder_path, filename)
        print("Saving model as", filename)
        joblib.dump(model, full_path)


class BaggingClassifier:

    def __init__(self, base_model, X_train, y_train, X_test, y_test, n_straps=100, k_fold=5):

        self.base_model = base_model
        self.n_straps = n_straps
        self.k_fold = k_fold
        self.models = []
        self.X_train, self.y_train = X_train, y_train
        self.X_test = X_test
        self.y_test = y_test
        self.accuracy_scores = []
        self.sensitivity_scores = []
        self.specificity_scores = []
        self.avg_accuracy = None
        self.avg_sensitivity = None
        self.avg_specificity = None

    def examine_bagging(self):

        kfold = KFold(n_splits=self.k_fold, shuffle=True)
        for train_index, val_index in kfold.split(self.X_train):
            # Generate n_straps samples and train the models for the current fold
            self.models = []
            for _ in range(self.n_straps):
                # Create bootstrap sample with the available indices of the fold
                bootstrap_indices = np.random.choice(train_index, size=len(self.X_train[train_index]), replace=True)
                X_bootstrap = self.X_train[bootstrap_indices]
                y_bootstrap = self.y_train[bootstrap_indices]

                # Train base model on bootstrap sample
                model = clone(self.base_model)
                model.fit(X_bootstrap, y_bootstrap)
                self.models.append(model)

            y_pred = self.predict(self.X_train[val_index])

            self.cm = ConfusionMatrix(actual_vector=list(self.y_train[val_index]), predict_vector=list(y_pred))
            self.accuracy_scores.append(accuracy_score(self.y_train[val_index], y_pred))
            print(self.cm)
            self.sensitivity_scores.append(float(self.cm.TPR_Macro))
            self.specificity_scores.append(float(self.cm.TNR_Macro))

        self.avg_accuracy = sum(self.accuracy_scores) / len(self.accuracy_scores)
        self.avg_sensitivity = sum(self.sensitivity_scores) / len(self.sensitivity_scores)
        self.avg_specificity = sum(self.specificity_scores) / len(self.specificity_scores)

        return self.avg_accuracy, self.avg_sensitivity, self.avg_specificity

    def predict(self, X):

        # Aggregate predictions from all models
        predictions = np.zeros((len(X), self.n_straps))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)

        # Use majority voting to determine final prediction
        final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
        return final_predictions

    def evaluate(self, results_dict):

        self.y_pred = self.predict(self.X_test)
        self.cm = ConfusionMatrix(actual_vector=list(self.y_test), predict_vector=list(self.y_pred))
        self.accuracy_scores = accuracy_score(self.y_test, self.y_pred)
        self.sensitivity_scores = float(self.cm.TPR_Macro)
        self.specificity_scores = float(self.cm.TNR_Macro)
        self.f1_score_value = f1_score(self.y_test, self.y_pred, average='macro')  # F1-score

        results_dict['Bagging']['Accuracy'] = float(self.accuracy_scores)
        results_dict['Bagging']['Sensitivity'] = float(self.sensitivity_scores)
        results_dict['Bagging']['Specificity'] = float(self.specificity_scores)
        results_dict['Bagging']['F1_score'] = float(self.f1_score_value)  # Store F1-score in results

        print(f"Accuracy: {self.accuracy_scores}, Sensitivity: {self.sensitivity_scores}, "
              f"Specificity: {self.specificity_scores}, F1-score: {self.f1_score_value}")


class ModelTester:
    def __init__(self, builder, label_encoder, test_data_encoded, output_csv):
        """
        Initialize with the best model, PCA, scaler, and preprocessing configurations.
        """
        self.best_model = builder.best_model
        self.label_encoder = label_encoder
        self.test_data_encoded = test_data_encoded
        self.output_csv = output_csv

    def predict(self):

        # Make predictions
        predictions = self.best_model.predict(self.test_data_encoded)

        # Decode predicted genre numbers back to genre names
        predicted_genres = self.label_encoder.inverse_transform(predictions)

        # Add predictions to the original data
        self.test_data_encoded['predicted_genre'] = predicted_genres

        # Save to CSV
        self.test_data_encoded.to_csv(self.output_csv, index=False)
        print(f"Predictions saved to {self.output_csv}")

# %% 1- Pre Processing and EDA

filename = 'data/train.txt'
column_names = ['title', 'language', 'genre', 'director', 'plot']
glove_file_path = 'Glove/glove.6B.100d.txt'

# Load the data
data_loader = DataManipulator(filename, column_names ,'genre')

test_no_labels_filename = 'data/test_no_labels.txt'
test_no_labels_column_names = ['title', 'language', 'director', 'plot']
test_no_labels_data = pd.read_csv(test_no_labels_filename, sep='\t', names=test_no_labels_column_names)

# Preprocess the data
data_processing = DataProcessing(data_loader)
data_processing.clean_text()
test_data_preprocessed = data_processing.clean_text(test_no_labels_data)

data_loader.data.to_csv('data/movie_data_preprocessed.csv', index=False)
test_data_preprocessed.to_csv('data/movie_data_no_labels_preprocessed.csv', index=False)

# Visualize the data
#data_visualization = DataVisualization(data_loader, ['pie', 'bar', 'hist', 'wordcloud'])
#data_visualization.plots(['pie', 'bar', 'hist', 'wordcloud'])

# %% 2- Feature Creation

# Initialize the FeatureCreation class with the data
feature_creator = FeatureCreation()
# Create features
print("\nFeatures Created:\n")
feature_creator.create_features(data_loader.data)
feature_creator.create_features(test_data_preprocessed)

# Implement some kind of data visualization

data_loader.data.to_csv('data/movie_data_featurecreation.csv', index=False)
test_data_preprocessed.to_csv('data/movie_data_no_labels_featurecreation.csv', index=False)

# %% 3- Data Splitting

X_raw = data_loader.data.drop(columns=[data_loader.target])
y_raw = data_loader.data[data_loader.target]

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# %% 4- Embeddings and Data Splitting

data_embeddings = Embeddings(glove_file_path)
X_train_glove, X_test_glove, scaler_glove, label_encoders_X_glove = data_embeddings.embedding_X(X_train_raw,X_test_raw, 'glove')
X_train_bert, X_test_bert, scaler_bert, label_encoders_X_bert = data_embeddings.embedding_X(X_train_raw,X_test_raw, 'bert')

y_train, y_test = data_embeddings.encode_target(y_train_raw, y_test_raw)

y_test_no_labels_glove = data_embeddings.embedding_test(test_data_preprocessed, scaler_glove, label_encoders_X_glove, 'glove')
y_test_no_labels_bert = data_embeddings.embedding_test(test_data_preprocessed, scaler_bert, label_encoders_X_bert, 'bert')

X_train_glove, X_validation_glove, y_train_glove, y_validation_glove = train_test_split(X_train_glove, y_train, test_size=0.2, random_state=42)
X_train_bert, X_validation_bert, y_train_bert, y_validation_bert = train_test_split(X_train_bert, y_train, test_size=0.2, random_state=42)


np.save('data/vectors/X_train_glove', X_train_glove)
np.save('data/vectors/X_test_glove', X_test_glove)
np.save('data/vectors/X_validation_glove', X_validation_glove)
np.save('data/vectors/y_train_glove', y_train_glove)
np.save('data/vectors/y_test_glove', y_test)
np.save('data/vectors/y_validation_glove', y_validation_glove)

np.save('data/vectors/X_train_bert', X_train_bert)
np.save('data/vectors/X_test_bert', X_test_bert)
np.save('data/vectors/X_validation_bert', X_validation_bert)
np.save('data/vectors/y_train_bert', y_train_bert)
np.save('data/vectors/y_validation_bert', y_validation_bert)

np.save('data/vectors/y_test_no_labels_glove', y_test_no_labels_glove)
np.save('data/vectors/y_test_no_labels_bert', y_test_no_labels_bert)

# Save dictionarie label_encoders_X_glove and label_encoders_X_bert
joblib.dump(label_encoders_X_glove, 'data/vectors/label_encoders_X_glove.pkl')
joblib.dump(label_encoders_X_bert, 'data/vectors/label_encoders_X_bert.pkl')

# X_train_glove = np.load('data/vectors/X_train_glove.npy')
# X_test_glove = np.load('data/vectors/X_test_glove.npy')
# X_validation_glove = np.load('data/vectors/X_validation_glove.npy')
# y_train_glove = np.load('data/vectors/y_train_glove.npy')
# y_test = np.load('data/vectors/y_test_glove.npy')
# y_validation_glove = np.load('data/vectors/y_validation_glove.npy')
#
# X_train_bert = np.load('data/vectors/X_train_bert.npy')
# X_test_bert = np.load('data/vectors/X_test_bert.npy')
# X_validation_bert = np.load('data/vectors/X_validation_bert.npy')
# y_train_bert = np.load('data/vectors/y_train_bert.npy')
# y_validation_bert = np.load('data/vectors/y_validation_bert.npy')
#
# y_test_no_labels_glove = np.load('data/vectors/y_test_no_labels_glove.npy')
# y_test_no_labels_bert = np.load('data/vectors/y_test_no_labels_bert.npy')
#
# # Load dictionarie label_encoders_X_glove and label_encoders_X_bert
# label_encoders_X_glove = joblib.load('data/vectors/label_encoders_X_glove.pkl')
# label_encoders_X_bert = joblib.load('data/vectors/label_encoders_X_bert.pkl')

evaluator = FeatureEvaluator(X_train, y_train)
evaluator.evaluate_importance()
evaluator.plot_importance()

top_features = evaluator.get_top_features(n=10)
print("Top 10 features:", top_features)

# %% 5- Model Building

# Define the results dictionary
results_dict = {
    "LogisticRegression with Glove Embeddings": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0, "F1_score": 0.0},
    "RandomForest with Glove Embeddings": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0, "F1_score": 0.0},
    "SVM with Glove Embeddings": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0, "F1_score": 0.0},
    "MLP with Glove Embeddings": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0, "F1_score": 0.0},
    "LogisticRegression with BERT Embeddings": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0, "F1_score": 0.0},
    "RandomForest with BERT Embeddings": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0, "F1_score": 0.0},
    "SVM with BERT Embeddings": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0, "F1_score": 0.0},
    "MLP with BERT Embeddings": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0, "F1_score": 0.0},
    "Bagging": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0, "F1_score": 0.0}
}

# Define the model dictionary for optimization
models_dict_glove = {
    "LogisticRegression": {
        "model": LogisticRegression,"C_values": (0.01, 0.1, 1.0, 10.0),"penalty": (None, 'l2')},
    "RandomForest": {
        "model": RandomForestClassifier,"n_estimators": (100, 200, 500),"max_depth": (10, 20, None),"min_samples_split": (2, 5, 10)},
    "SVM": {
        "model": SVC,"C_values": (0.1, 1, 10),"kernel": ('linear', 'rbf'),"gamma": ('scale', 'auto')},
    "MLP": {
        "model": MLPClassifier, "population_size": 5, "max_generations": 20, "layer_range": (50, 200), "activation": ("tanh", "logistic", "relu")}
}

# Initialize the ModelBuilding class with the training, testing and validation data
builder_glove = ModelBuilding(np.array(X_train_glove), np.array(y_train_glove), np.array(X_test_glove), np.array(y_test),
                              np.array(X_validation_glove), np.array(y_validation_glove), 'glove')

builder_bert = ModelBuilding(np.array(X_train_bert), np.array(y_train_bert), np.array(X_test_bert), np.array(y_test),
                             np.array(X_validation_bert), np.array(y_validation_bert), 'bert')

# Supervised Learning Algorithms
builder_glove.build_models("LogisticRegression", models_dict_glove, results_dict)
builder_glove.build_models("RandomForest", models_dict_glove, results_dict)
builder_glove.build_models("SVM", models_dict_glove, results_dict)
builder_glove.build_models("MLP", models_dict_glove, results_dict)

models_dict_bert = {
    "LogisticRegression": {
        "model": LogisticRegression,"C_values": (0.01, 0.1, 1.0, 10.0),"penalty": (None, 'l2')},
    "RandomForest": {
        "model": RandomForestClassifier,"n_estimators": (100, 200, 500),"max_depth": (10, 20, None),"min_samples_split": (2, 5, 10)},
    "SVM": {
        "model": SVC,"C_values": (0.1, 1, 10),"kernel": ('linear', 'rbf'),"gamma": ('scale', 'auto')},
    "MLP": {
        "model": MLPClassifier, "population_size": 5, "max_generations": 20, "layer_range": (50, 200), "activation": ("tanh", "logistic", "relu")}
}

builder_bert.build_models("LogisticRegression", models_dict_bert, results_dict)
builder_bert.build_models("RandomForest", models_dict_bert, results_dict)
builder_bert.build_models("SVM", models_dict_bert, results_dict)
builder_bert.build_models("MLP", models_dict_bert, results_dict)

# Serialize the builder object
with open('builder_glove.pkl', 'wb') as f:
    pickle.dump(builder_glove, f)
# Deserialize the builder object
with open('builder_glove.pkl', 'rb') as f:
    builder_glove = pickle.load(f)

# Serialize the builder object
with open('builder_bert.pkl', 'wb') as f:
    pickle.dump(builder_bert, f)
# Deserialize the builder object
with open('builder_bert.pkl', 'rb') as f:
    builder_bert = pickle.load(f)

# Name of the best model
best_model_name_glove = None
best_model_name_bert = None

# Check the best model
if builder_glove.best_model_checked == LogisticRegression:
    best_model_name_glove = 'LogisticRegression'
elif builder_glove.best_model_checked == RandomForestClassifier:
    best_model_name_glove = 'RandomForest'
elif builder_glove.best_model_checked == SVC:
    best_model_name_glove = 'SVM'
elif builder_glove.best_model_checked == MLPClassifier:
    best_model_name_glove = 'MLP'

# Check the best model
if builder_bert.best_model_checked == LogisticRegression:
    best_model_name_bert = 'LogisticRegression'
elif builder_bert.best_model_checked == RandomForestClassifier:
    best_model_name_bert = 'RandomForest'
elif builder_bert.best_model_checked == SVC:
    best_model_name_bert = 'SVM'
elif builder_bert.best_model_checked == MLPClassifier:
    best_model_name_bert = 'MLP'

print("\nBest model Glove:", best_model_name_glove)
print("Best model Glove parameters:", builder_glove.best_model_params_checked)

print("\nBest model Bert:", best_model_name_bert)
print("Best model Bert parameters:", builder_bert.best_model_params_checked)

# Comparação dos dois modelos
if builder_glove.best_score > builder_bert.best_score:
    builder = builder_glove
    X_train = X_train_glove
    y_train = y_train_glove
    X_test = X_test_glove
    label_encoder = label_encoders_X_glove

    print("\nGlove embeddings model is better\n")
else:
    builder = builder_bert
    X_train = X_train_bert
    y_train = y_train_bert
    X_test = X_test_bert
    label_encoder = label_encoders_X_bert

    print("\nBert embeddings model is better\n")

# Initialize the BaggingClassifier class with the best model
bagging = BaggingClassifier(builder.best_model_checked(**builder.best_model_params_checked), np.array(X_train),
                            np.array(y_train), np.array(X_test), np.array(y_test))
# Examine the bagging ensemble
bagging.examine_bagging()
# Evaluate the bagging ensemble on the test set
bagging.evaluate(results_dict)

# Assuming 'builder' contains the best model, PCA, and scaler
tester = ModelTester(builder, label_encoder, test_data_preprocessed, 'data/predictions.csv')
# Predict the target and save to CSV
tester.predict()
