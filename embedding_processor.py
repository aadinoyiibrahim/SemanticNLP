"""
 
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

        # Initialize the embedding helper
        self.embedding_helper = EmbeddingHelper()

        # Initialize the lemmatizer and stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.en_stopwords = set(stopwords.words("english"))

        # Define keyword variations
        self.keyword_variations = {
            "deep learning": [
                "deep learning",
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
                "transformer",
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
        # Remove non-alphabetic characters and lower the case
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
        # Prepare the 'titleAbstractNew' column
        self.df_cleaned["titleAbstractNew"] = (
            self.df_cleaned["Title"] + " " + self.df_cleaned["Abstract"]
        )
        self.df_cleaned["titleAbstractNew"] = self.df_cleaned["titleAbstractNew"].apply(
            self.lemmatize_text
        )

        # Generate embeddings using the helper method
        self.df_cleaned["titleAbstractNew_Embedding"] = self.df_cleaned[
            "titleAbstractNew"
        ].apply(self.embedding_helper.get_embedding)

        # Convert the list of embeddings into a 2D NumPy array
        embeddings_matrix = np.vstack(
            self.df_cleaned["titleAbstractNew_Embedding"].values
        )

        # Create a query vector from the keywords and ensure it's 2D
        query_vector = self.embedding_helper.get_embedding(
            " ".join(self.keyword_variations.keys())
        ).reshape(1, -1)

        # Calculate weighted cosine similarity
        weighted_similarities = self.calculate_weighted_similarity(
            embeddings_matrix, query_vector
        )

        # Assign the results back to the DataFrame
        self.df_cleaned["titleAbstractNew_Similarity"] = weighted_similarities.flatten()
        return self.df_cleaned

    def calculate_weighted_similarity(self, embeddings, query_vector):
        # Calculate cosine similarity
        similarities = cosine_similarity(embeddings, query_vector)

        # Apply penalty for papers that do not contain relevant keywords
        for i in range(similarities.shape[0]):
            paper_text = self.df_cleaned["titleAbstractNew"].iloc[i]
            if not self.embedding_helper.contains_keywords(
                paper_text, self.keyword_variations
            ):
                similarities[i] *= 0.1  # Apply a penalty for irrelevant papers
        return similarities

    def classify_paper(self, text):
        for method, keywords in self.keyword_variations.items():
            if any(keyword in text for keyword in keywords):
                return method  # Return the first matching method
        return "Others"  # If no keywords match

    def classify_relevant_methods(self):
        # Apply classification function to each paper
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


# Example usage
if __name__ == "__main__":
    # Provide the path to your CSV file here
    data_path = "data.csv"  # Adjust the path as needed

    # Initialize the EmbeddingProcessor with the data path
    embedding_processor = EmbeddingProcessor(data_path)

    # Process embeddings and get results
    results = embedding_processor.process_embeddings()

    # Display results
    print(results.head(10))

    # Plot the distribution of methods
    embedding_processor.plot_method_distribution()
