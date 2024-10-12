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
from sklearn.base import clone
from textblob import TextBlob

# Download necessary resources for NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import wordnet

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

    def __init__(self, data_loader, glove_file_path, reduce_dims=True, num_components=50):
        self.data_loader = data_loader
        self.stop_words = set(stopwords.words('english'))
        self.embeddings_index = self._load_glove_embeddings(glove_file_path)
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

    def _plot_to_embedding(self, text):
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

        if word_count == 0:
            print(f"No valid words found for text: {text}")  # Debugging statement
            return np.nan  # or return a zero vector

        return plot_vector

    def embeddings(self, X_train_augmented, y_train_augmented, X_test_raw, y_test_raw):
        """Create dense vector representations for both train and test data and return encoded X and y."""

        # ----- Processing X_train_augmented -----

        # Create plot embeddings for X_train_augmented
        X_train_augmented['plot_embedding'] = X_train_augmented['plot'].apply(self._plot_to_embedding)

        # Stack the plot embeddings into a matrix
        plot_embeddings_train = np.vstack(X_train_augmented['plot_embedding'].values)

        # Apply PCA to reduce dimensionality
        if self.reduce_dims:
            pca = PCA(n_components=self.num_components)
            plot_embeddings_train = pca.fit_transform(plot_embeddings_train)
            print(f"Reduced training plot embeddings to {self.num_components} dimensions using PCA.")

        # Convert plot embeddings into a DataFrame
        plot_embeddings_train_df = pd.DataFrame(plot_embeddings_train,
                                                columns=[f'embedding_{i}' for i in
                                                         range(plot_embeddings_train.shape[1])])

        # Drop unnecessary columns and concatenate the plot embeddings
        X_train_processed = pd.concat([X_train_augmented.drop(columns=['plot', 'title', 'plot_embedding']),
                                       plot_embeddings_train_df], axis=1)

        # Label encode categorical features in X_train_augmented
        for col in ['language', 'director']:
            le = LabelEncoder()
            X_train_processed[col] = le.fit_transform(X_train_processed[col].astype(str))

        # ----- Processing X_test_raw -----

        # Create plot embeddings for X_test_raw
        X_test_raw['plot_embedding'] = X_test_raw['plot'].apply(self._plot_to_embedding)

        # Stack the plot embeddings into a matrix
        plot_embeddings_test = np.vstack(X_test_raw['plot_embedding'].values)

        # Apply PCA to reduce dimensionality
        if self.reduce_dims:
            plot_embeddings_test = pca.transform(plot_embeddings_test)  # Use the same PCA model from training
            print(f"Reduced test plot embeddings to {self.num_components} dimensions using PCA.")

        # Convert plot embeddings into a DataFrame
        plot_embeddings_test_df = pd.DataFrame(plot_embeddings_test,
                                               columns=[f'embedding_{i}' for i in range(plot_embeddings_test.shape[1])])

        # Reset the index of the test DataFrame to avoid any alignment issues
        X_test_raw.reset_index(drop=True, inplace=True)
        plot_embeddings_test_df.reset_index(drop=True, inplace=True)

        # Drop unnecessary columns and concatenate the plot embeddings
        X_test_processed = pd.concat([X_test_raw.drop(columns=['plot', 'title', 'plot_embedding']),
                                      plot_embeddings_test_df], axis=1)

        # Label encode categorical features in X_test_raw
        for col in ['language', 'director']:
            le = LabelEncoder()
            X_test_processed[col] = le.fit_transform(X_test_processed[col].astype(str))

        # Check for NaN values in processed test features
        for column in X_test_processed.columns:
            nan_count = X_test_processed[column].isnull().sum()
            if nan_count > 0:
                print(f"NaN values found in test column '{column}': {nan_count}")

        # ----- Encoding target variables -----
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_augmented)
        y_test_encoded = label_encoder.fit_transform(y_test_raw)

        # ----- Standardizing feature matrices -----
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_processed)
        X_test_scaled = scaler.transform(X_test_processed)

        print("Dense embeddings created, reduced, and concatenated with other features for both training and test sets.")
        return X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded

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

    def create_text_based_features(self, data):

        # Create Plot Length Feature
        data['plot_length'] = data['plot'].apply(lambda x: len(str(x).split()))
        print("Created plot_length feature\n")
        # Create Average Word Length Feature
        data['avg_word_length'] = data['plot'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]))
        print("Created avg_word_length feature\n")
        # Create Unique Word Count Feature
        data['unique_word_count'] = data['plot'].apply(
            lambda x: len(set(str(x).split())))
        print("Created unique_word_count feature\n")
        # Create Sentiment Polarity Feature
        data['sentiment_polarity'] = data['plot'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity)
        print("Created sentiment_polarity feature\n")

class DataAugmentation:

    def __init__(self):
        pass

    def get_synonyms(self, word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        return set(synonyms)

    def synonym_replacement(self, text, n_replacements):
        words = word_tokenize(text)
        new_words = words.copy()

        # List of indices where replacements will happen
        indices_to_replace = random.sample(range(len(words)), min(n_replacements, len(words)))
        successful_replacements = 0
        replacements_log = []

        for idx in indices_to_replace:
            word = words[idx]
            synonyms = self.get_synonyms(word)
            if synonyms:
                # Replace the word with a random synonym if available
                synonym = random.choice(list(synonyms))
                new_words[idx] = synonym
                successful_replacements += 1
                replacements_log.append((word, synonym))

        new_plot = ' '.join(new_words)
        return new_plot

    def augment_data(self, data, n_augmentations):
        original_row_count = len(data)
        new_rows = []

        # Iterate over each row in the dataset
        for _, row in data.iterrows():
            original_row = row.to_dict()  # Convert the row to a dictionary

            # Generate n_augmentations new rows with modified plots
            for _ in range(n_augmentations):
                new_row = original_row.copy()  # Keep other columns the same
                new_row['plot'] = self.synonym_replacement(row['plot'], n_replacements=50)
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

        # Check for NaN values in the augmented data
        nan_values = augmented_data.isnull().sum()
        nan_columns = nan_values[nan_values > 0]  # Filter columns with NaN values
        if nan_columns.empty:
            print("No NaN values found in the augmented data.")
        else:
            print(f"NaN values found in the following columns:\n{nan_columns}")

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

        results_dict['Bagging']['Accuracy'] = float(self.accuracy_scores)
        results_dict['Bagging']['Sensitivity'] = float(self.sensitivity_scores)
        results_dict['Bagging']['Specificity'] = float(self.specificity_scores)

        print(f"Accuracy: {self.accuracy_scores}, Sensitivity: {self.sensitivity_scores}, Specificity: {self.specificity_scores}")

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

# %% 2- Data Augmentation and Data Splitting

X_raw = data_loader.data.drop(columns=[data_loader.target])
y_raw = data_loader.data[data_loader.target]

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

data_augmentation = DataAugmentation()
X_train_augmented = data_augmentation.augment_data(X_train_raw, n_augmentations=1)
y_train_augmented = pd.concat([y_train_raw] * (X_train_augmented.shape[0] // X_train_raw.shape[0]), ignore_index=True)

# %% 3- Feature Creation

# Initialize the FeatureCreation class with the data
feature_creator = FeatureCreation()

print("\nFeatures Created:\n")
# Create text-based features
feature_creator.create_text_based_features(X_train_augmented)
feature_creator.create_text_based_features(X_test_raw)

# %% 4- Embeddings and Data Splitting

X_train, y_train, X_test, y_test = data_preprocessing.embeddings(X_train_augmented, y_train_augmented, X_test_raw, y_test_raw)

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
    "SVM": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0},
    "Bagging": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0},
    "AdaBoost": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0}
}

# Initialize the ModelBuilding class with the training, testing and validation data
builder = ModelBuilding(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test),
                        np.array(X_validation), np.array(y_validation))

# Supervised Learning Algorithms
#builder.build_models("LogisticRegression", models_dict, results_dict)
#builder.build_models("RandomForest", models_dict, results_dict)
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

# Initialize the BaggingClassifier class with the best model
bagging = BaggingClassifier(builder.best_model_checked(**builder.best_model_params_checked), np.array(X_train),
                            np.array(y_train), np.array(X_test), np.array(y_test))
# Examine the bagging ensemble
bagging.examine_bagging()
# Evaluate the bagging ensemble on the test set
bagging.evaluate(results_dict)
