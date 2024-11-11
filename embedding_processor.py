"""
 contains the EmbeddingProcessor class that 
 processes the embeddings for the papers and 
 compute the similarity between the papers and the keywords. 
 It also classifies the papers into relevant methods.
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from methods import EmbeddingHelper
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download("all")


class EmbeddingProcessor:
    def __init__(self, data_path):
        # Load the dataset
        self.df = pd.read_csv(data_path)
        self.df_cleaned = self.df.dropna(subset=["Abstract"])
        self.embedding_helper = EmbeddingHelper()  # Initialize

        self.lemmatizer = WordNetLemmatizer()  # Initialize
        self.en_stopwords = set(stopwords.words("english"))

        # keyword variations
        self.keyword_variations = {
            "deep learning": [
                "deep learning",
                "deeplearning",
                "deep neural networks",
                "neural network",
                "artificial neural network",
                "feedforward neural network",
                "neural net algorithm",
                "multilayer perceptron",
                "convolutional neural network",
                "cnn",
                "recurrent neural network",
                "rnn",
                "long short-term memory network",
                "lstm",
                # "transformer",
                "transformer models",
                "transformer-based model",
                "self-attention models",
                "attention-based neural networks",
            ],
            "computer vision": [
                "computer vision",
                "vision model",
                "image processing",
                "vision algorithms",
                "object recognition",
                "scene understanding",
            ],
            "natural language processing": [
                "natural language processing",
                "text mining",
                "nlp",
                "computational linguistics",
                "language processing",
                "text analytics",
                "textual data analysis",
                "text data analysis",
                "text analysis",
                "speech and language technology",
                "language modeling",
                "computational semantics",
            ],
            "large language model": [
                "large language model",
                "pretrained language model",
                "generative language model",
                "foundation model",
                "state-of-the-art language model",
            ],
        }

    def lemmatize_text(self, text):
        """
        Lemmatize the text and remove stopwords

        param   text: input text to lemmatize

        return: lemmatized text
        """
        text = re.sub(r"[^A-Za-z1-9 ]", "", text)
        text = text.lower()
        tokens = word_tokenize(text)
        clean_text = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.en_stopwords
        ]
        return " ".join(clean_text)

    def process_embeddings(self):
        """
        Process the embeddings for the papers and calculate the similarity
        between the papers and the keywords

        """

        self.df_cleaned["titleAbstractNew"] = (
            self.df_cleaned["Title"] + " " + self.df_cleaned["Abstract"]
        )
        self.df_cleaned["titleAbstractNew"] = self.df_cleaned["titleAbstractNew"].apply(
            self.lemmatize_text
        )

        self.df_cleaned["titleAbstractNew_Embedding"] = self.df_cleaned[
            "titleAbstractNew"
        ].apply(self.embedding_helper.get_embedding)

        # Convert embeddings into a 2D NumPy array
        embeddings_matrix = np.vstack(
            self.df_cleaned["titleAbstractNew_Embedding"].values
        )

        query_vector = self.embedding_helper.get_embedding(
            " ".join(self.keyword_variations.keys())
        ).reshape(1, -1)

        weighted_similarities = self.calculate_weighted_similarity(
            embeddings_matrix, query_vector
        )

        # Assign the results back to the DataFrame
        self.df_cleaned["titleAbstractNew_Similarity"] = weighted_similarities.flatten()
        return self.df_cleaned

    def calculate_weighted_similarity(self, embeddings, query_vector):
        """
        Calculate the cosine similarity between the embeddings and the query vector
        param
            embeddings: 2D NumPy array of embeddings
            query_vector: 2D NumPy array of the query vector

        return: 1D NumPy array of weighted similarities
        """

        similarities = cosine_similarity(embeddings, query_vector)

        for i in range(similarities.shape[0]):
            paper_text = self.df_cleaned["titleAbstractNew"].iloc[i]
            if not self.embedding_helper.contains_keywords(
                paper_text, self.keyword_variations
            ):
                similarities[i] *= 0.1  # penalize
        return similarities

    def classify_paper(self, text):
        """
        Classify the paper based on the keywords present in the text
        param
            text: input text to classify
        return: method type
        """
        for method, keywords in self.keyword_variations.items():
            if any(keyword in text for keyword in keywords):
                return method  # Return the first matching method
        return "Others"

    def classify_relevant_methods(self):
        """
        Classify the papers into relevant methods
        """
        self.df_cleaned["method_type"] = self.df_cleaned["titleAbstractNew"].apply(
            self.classify_paper
        )
        return (
            self.df_cleaned[["Title", "method_type"]]
            .groupby("method_type")
            .size()
            .reset_index(name="count")
        )

    def classify_method(self, text):
        """
        Classify the method based on the keywords present in the text
        param   text: input text to classify
        return: method type
        """
        normalized_text = self.embedding_helper.normalize_text(
            text, self.keyword_variations
        )
        has_text_mining = any(
            keyword in normalized_text
            for keyword in self.keyword_variations["natural language processing"]
        )
        has_computer_vision = any(
            keyword in normalized_text
            for keyword in self.keyword_variations["computer vision"]
        )

        if has_text_mining and has_computer_vision:
            return "both"
        elif has_text_mining:
            return "text mining"
        elif has_computer_vision:
            return "computer vision"
        else:
            return "other"

    def classify_relevant_methods(self):
        """
        Classify the papers into relevant methods
        """
        self.df_cleaned["method_group"] = self.df_cleaned["titleAbstractNew"].apply(
            self.classify_method
        )
        return (
            self.df_cleaned[["Title", "method_group"]]
            .groupby("method_group")
            .size()
            .reset_index(name="count")
        )

    def plot_method_distribution(self):
        methods_report = self.classify_relevant_methods()
        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            methods_report["method_group"], methods_report["count"], color="blue"
        )
        plt.xlabel("Method")
        plt.ylabel("Count")
        plt.title("Distribution of Methods in Relevant Papers")
        plt.xticks(rotation=45)
        plt.yscale("log")

        total_count = methods_report["count"].sum()
        # Adding value labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            percentage = (yval / total_count) * 100
            label = f"{int(yval)} ({percentage:.1f}%)"
            plt.text(
                bar.get_x() + bar.get_width() / 2, yval, label, ha="center", va="bottom"
            )

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    data_path = "data.csv"

    embedding_processor = EmbeddingProcessor(data_path)
