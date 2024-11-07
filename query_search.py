import requests
import pandas as pd
from bs4 import BeautifulSoup
import os

# Step 1: Define search queries
search_queries = {
    "virology_neural_networks": '(((virology) OR (epidemiology)) AND (("neural network") OR ("artificial neural network") OR ("machine learning model") OR ("feedforward neural network") OR ("neural net algorithm") OR ("multilayer perceptron") OR ("convolutional neural network") OR ("recurrent neural network") OR ("long short-term memory network") OR ("CNN") OR ("GRNN") OR ("RNN") OR ("LSTM")))',
    "virology_deep_learning": '(((virology) OR (epidemiology)) AND (("deep learning") OR ("deep neural networks")))',
    "virology_computer_vision": '(((virology) OR (epidemiology)) AND (("computer vision") OR ("vision model") OR ("image processing") OR ("vision algorithms") OR ("computer graphics and vision") OR ("object recognition") OR ("scene understanding")))',
    "virology_nlp": '(((virology) OR (epidemiology)) AND (("natural language processing") OR ("text mining") OR (NLP) OR ("computational linguistics") OR ("language processing") OR ("text analytics") OR ("textual data analysis") OR ("text data analysis") OR ("text analysis") OR ("speech and language technology") OR ("language modeling") OR ("computational semantics")))',
    "virology_generative_ai": '(((virology) OR (epidemiology)) AND (("generative artificial intelligence") OR ("generative AI") OR ("generative deep learning") OR ("generative models")))',
    "virology_transformer": '(((virology) OR (epidemiology)) AND (("transformer models") OR ("self-attention models") OR ("transformer architecture") OR (transformer) OR ("attention-based neural networks") OR ("transformer networks") OR ("sequence-to-sequence models")))',
    "virology_llm": '(((virology) OR (epidemiology)) AND (("large language model") OR (llm) OR ("transformer-based model") OR ("pretrained language model") OR ("generative language model") OR ("foundation model") OR ("state-of-the-art language model")))',
    "virology_multimodal": '(((virology) OR (epidemiology)) AND (("multimodal model") OR ("multimodal neural network") OR ("vision transformer") OR ("diffusion model") OR ("generative diffusion model") OR ("diffusion-based generative model") OR ("continuous diffusion model")))',
}


# Function to perform search query and get results
def fetch_pubmed_data(query):
    base_url = "https://pubmed.ncbi.nlm.nih.gov/"
    search_url = f"{base_url}?term={requests.utils.quote(query)}&format=abstract"

    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract article metadata
    articles = []
    for article in soup.find_all("article"):
        title = article.find("h1").text.strip()
        pmid = article.find("meta", {"name": "pub-id"}).get("content")
        articles.append({"title": title, "pmid": pmid})

    return articles


# Step 1: Query for each technology and save results
all_articles = []
for key, query in search_queries.items():
    articles = fetch_pubmed_data(query)
    all_articles.extend(articles)

# Step 2: Convert to DataFrame and de-duplicate based on PMID
df = pd.DataFrame(all_articles)
df.drop_duplicates(subset="pmid", inplace=True)  # De-duplicate based on PMID

# Save merged results to a CSV file
merged_filename = "merged_pubmed_results.csv"
df.to_csv(merged_filename, index=False)
print(f"Merged results saved to {merged_filename} with {len(df)} unique records.")


# Step 3: Fetch abstracts for unique articles
def fetch_abstract(pmid):
    abstract_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    response = requests.get(abstract_url)
    soup = BeautifulSoup(response.content, "html.parser")
    abstract = (
        soup.find("div", class_="abstract-content").get_text(strip=True)
        if soup.find("div", class_="abstract-content")
        else "Abstract not found"
    )
    return abstract


# Fetch abstracts for all unique PMIDs
df["abstract"] = df["pmid"].apply(fetch_abstract)

# Save final DataFrame with abstracts
final_filename = "pubmed_with_abstracts.csv"
df.to_csv(final_filename, index=False)
print(f"Final results with abstracts saved to {final_filename}.")
