"""
contains the methods to get the embeddings 
of the text and check if the text contains the keywords
"""

import re
import torch
from transformers import DistilBertTokenizer, DistilBertModel


class EmbeddingHelper:
    def __init__(self):
        # Load the DistilBERT tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move model to GPU if available

    def normalize_text(self, text, keyword_variations):
        """
        Normalize the text by converting to
        lowercase and replacing keyword variations with standard terms

        param
            text: input text to normalize
            keyword_variations: dictionary of standard keywords and their variations

        return: normalized text
        """
        text = text.lower()
        for standard, variations in keyword_variations.items():
            for variation in variations:
                text = re.sub(r"\b" + re.escape(variation) + r"\b", standard, text)
        return text

    def get_embedding(self, text):
        """
        Get the embeddings of the input text using DistilBERT model
        param
            text: input text to get embeddings

        return: 2D NumPy array of embeddings
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

    def contains_keywords(self, text, keyword_variations):
        """
        Check if the text contains any of the keywords
        param
            text: input text to check
            keyword_variations: dictionary of standard keywords and their variations

        return: True if any keyword is found, False otherwise
        """
        normalized_text = self.normalize_text(text, keyword_variations)
        return any(keyword in normalized_text for keyword in keyword_variations.keys())
