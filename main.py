import joblib
import pickle
import os
import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from docutils.parsers.rst.directives.misc import Class
from wordcloud import WordCloud
from pycm import ConfusionMatrix
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from textblob import TextBlob

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

    def __init__(self, data_loader, glove_file_path, reduce_dims=True, num_components=75):
        self.data_loader = data_loader
        self.stop_words = set(stopwords.words('english'))
        self.embeddings_index = self.load_glove_embeddings(glove_file_path)
        self.reduce_dims = reduce_dims
        self.num_components = num_components

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

    def load_glove_embeddings(self, file_path):
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

    def plot_to_embedding(self, text):
        """Convert a plot of text into a dense vector representation using GloVe."""
        words = text.split()
        embedding_dim = 100  # Adjust based on the GloVe vector size
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

    def embeddings(self):
        """Create dense vector representations for each plot and return X and y."""
        # Create plot embeddings
        self.data_loader.data['plot_embedding'] = self.data_loader.data['plot'].apply(self.plot_to_embedding)

        # Stack the plot embeddings into a matrix
        plot_embeddings = np.vstack(self.data_loader.data['plot_embedding'].values)

        # Optional: Apply PCA to reduce the dimensions of embeddings if reduce_dims is True
        if self.reduce_dims:
            pca = PCA(n_components=self.num_components)
            plot_embeddings = pca.fit_transform(plot_embeddings)
            print(f"Reduced plot embeddings to {self.num_components} dimensions using PCA.")

        # Convert to a DataFrame and use as a single column
        plot_embeddings_df = pd.DataFrame(plot_embeddings,
                                          columns=[f'embedding_{i}' for i in range(plot_embeddings.shape[1])])

        # Drop unnecessary columns and concatenate the reduced GloVe vectors
        X = pd.concat([self.data_loader.data.drop(columns=['plot', 'title', 'plot_embedding', 'genre']),
                       plot_embeddings_df], axis=1)

        # Instead of one-hot encoding, use Label Encoding for categorical variables to reduce dimensions
        for col in ['language', 'director']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Convert column names to strings
        X.columns = X.columns.astype(str)

        # Encode the target variable
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(self.data_loader.data['genre'])  # Encode genres

        # Standardize the feature matrix for better performance with SVM and other models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print("Dense embeddings created, reduced (if enabled), and concatenated with other features.")
        return X_scaled, y

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

class GloveEmbeddings:

    def __init__(self, data_loader, glove_file_path):

        self.data_loader = data_loader
        self.embeddings_index = self.load_glove_embeddings(glove_file_path)

    def load_glove_embeddings(self, file_path):
        """
        Load GloVe pre-trained embeddings into a dictionary.
        """
        embeddings_index = {}
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print(f"Loaded {len(embeddings_index)} word vectors from GloVe.")
        return embeddings_index

    def plot_to_embedding(self, text):
        """
        Convert a plot of text into a dense vector representation using GloVe.
        """
        # Split the plot into words
        words = text.split()

        # Initialize a vector to accumulate word vectors
        embedding_dim = 100  # Adjust based on the GloVe vector size you are using
        plot_vector = np.zeros((embedding_dim,))
        word_count = 0

        for word in words:
            word_vector = self.embeddings_index.get(word.lower())
            if word_vector is not None:
                plot_vector += word_vector
                word_count += 1

        # If the plot contains valid words with vectors, return the average vector
        if word_count > 0:
            plot_vector /= word_count

        return plot_vector

    def create_dense_vectors(self):
        """
        Create dense vector representations for each plot.
        """
        # Apply the plot_to_embedding function to every plot in the dataset
        self.data_loader.data['plot_embedding'] = self.data_loader.data['plot'].apply(self.plot_to_embedding)

        # Convert the plot embedding column to separate columns for each embedding dimension
        plot_embeddings = np.vstack(self.data_loader.data['plot_embedding'].values)
        plot_embeddings_df = pd.DataFrame(plot_embeddings,
                                          columns=[f'embedding_{i}' for i in range(plot_embeddings.shape[1])])

        # Concatenate the dense embeddings with the original data
        self.data_loader.data = pd.concat([self.data_loader.data, plot_embeddings_df], axis=1)

        print("Created dense vector representations for each plot using GloVe.")

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

class CrossValidator:

    def __init__(self, k=5):

        self.k = k
        self.kf = KFold(n_splits=k, shuffle=True)
        self.cm = None
        self.accuracy_scores = []
        self.sensitivity_scores = []
        self.specificity_scores = []

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

        avg_accuracy = sum(self.accuracy_scores) / len(self.accuracy_scores)
        avg_sensitivity = sum(self.sensitivity_scores) / len(self.sensitivity_scores)
        avg_specificity = sum(self.specificity_scores) / len(self.specificity_scores)

        return avg_accuracy, avg_sensitivity, avg_specificity

    def evaluate_on_test_set(self, model, X_test, y_test):

        y_pred = model.predict(X_test)
        cm = ConfusionMatrix(actual_vector=list(y_test), predict_vector=list(y_pred))
        accuracy = cm.Overall_ACC
        sensitivity = cm.TPR_Macro
        specificity = cm.TNR_Macro
        return accuracy, sensitivity, specificity

class ModelBuilding:

    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val, k=5, save_all=True):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.k = k
        self.save_all = save_all
        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.best_score = -1
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
            else:
                raise ValueError("Model type is not supported.")

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
            avg_accuracy_cv, avg_sensitivity_cv, avg_specificity_cv = self.kf_cv.cross_validate(model_instance,
                                                                                                self.X_train,
                                                                                                self.y_train)

            print("Average accuracy during cross-validation:", avg_accuracy_cv)
            print("Average sensitivity during cross-validation:", avg_sensitivity_cv)
            print(f"Average specificity during cross-validation: {avg_specificity_cv}\n")

            print(f"\nPerformance of the {model_name} model on the Test set:")

            # Performance of the model on the test set
            accuracy_test, sensitivity_test, specificity_test = self.kf_cv.evaluate_on_test_set(model_instance,
                                                                                                self.X_test,
                                                                                                self.y_test)
            print("Test set accuracy:", accuracy_test)
            print("Test set sensitivity:", sensitivity_test)
            print(f"Test set specificity: {specificity_test}\n")

            # Store the results in the results dictionary
            results_dict[model_name]['Accuracy'] = float(accuracy_test)
            results_dict[model_name]['Sensitivity'] = float(sensitivity_test)
            results_dict[model_name]['Specificity'] = float(specificity_test)

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


# %% 1- Pre Processing and EDA

filename = 'data/train.txt'
column_names = ['title', 'language', 'genre', 'director', 'plot']
glove_file_path = 'Glove/glove.6B.100d.txt'

# Load the data
data_loader = DataManipulator(filename, column_names ,'genre')

# Preprocess the data
data_preprocessing = DataPreProcessing(data_loader, glove_file_path)

data_loader.data.to_csv('data/movie_data_preprocessed.csv', index=False)

# Visualize the data
#data_visualization = DataVisualization(data_loader, ['pie', 'bar', 'hist', 'wordcloud'])
#data_visualization.plots(['pie', 'bar', 'hist', 'wordcloud'])

# %% 2- Feature Creation

# Initialize the FeatureCreation class with the data
feature_creator = FeatureCreation(data_loader)

print("\nFeatures Created:\n")

# Create text-based features
feature_creator.create_text_based_features()
print(data_loader.data.info())

data_loader.data.to_csv('data/movie_data_featurecreation.csv', index=False)

# %% 3- Data Splits and Encoding

X, y = data_preprocessing.embeddings()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# %% 4- Model Building

# Define the model dictionary for optimization
models_dict = {
    "LogisticRegression": {
        "model": LogisticRegression,"C_values": (0.01, 0.1, 1.0, 10.0),"penalty": (None, 'l2')},
    "RandomForest": {
        "model": RandomForestClassifier,"n_estimators": (100, 200, 500),"max_depth": (10, 20, None),"min_samples_split": (2, 5, 10)},
    "SVM": {
        "model": SVC,"C_values": (0.1, 1, 10),"kernel": ('linear', 'rbf'),"gamma": ('scale', 'auto')}
}

# Define the results dictionary
results_dict = {
    "LogisticRegression": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0},
    "RandomForest": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0},
    "SVM": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0}
}

# Initialize the ModelBuilding class with the training, testing and validation data
builder = ModelBuilding(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test),
                        np.array(X_validation), np.array(y_validation))

# Supervised Learning Algorithms
builder.build_models("LogisticRegression", models_dict, results_dict)
builder.build_models("RandomForest", models_dict, results_dict)
builder.build_models("SVM", models_dict, results_dict)

# Serialize the builder object
with open('builder.pkl', 'wb') as f:
    pickle.dump(builder, f)

# Deserialize the builder object
with open('builder.pkl', 'rb') as f:
    builder = pickle.load(f)

# Name of the best model
best_model_name = None

# Check the best model
if builder.best_model_checked == LogisticRegression:
    best_model_name = 'LogisticRegression'
elif builder.best_model_checked == RandomForestClassifier:
    best_model_name = 'RandomForest'
elif builder.best_model_checked == SVC:
    best_model_name = 'SVM'

print("\nBest model:", best_model_name)
print("Best model parameters:", builder.best_model_params_checked)
