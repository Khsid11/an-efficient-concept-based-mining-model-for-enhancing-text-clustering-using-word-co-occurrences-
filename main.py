from flask import Flask, render_template, request, send_file
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load spaCy model with word vectors
nlp = spacy.load("en_core_web_md")  # Using spaCy's medium-sized word vectors

# Hardcoded dataset path
DATASET_FOLDER = 'C:\\Users\\DELL\\Desktop\\moon\\dataset'

# Function to load data from folders
def load_data(folder_path):
    data = []
    labels = []

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        data.append(content)
                        labels.append(label)

    return data, labels

# Function to preprocess text data using spaCy and word embeddings
def preprocess_text_data(data):
    processed_data = []
    for text in data:
        doc = nlp(text)
        processed_text = ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
        processed_data.append(processed_text)
    return processed_data

# Function to calculate word co-occurrence matrix
def calculate_co_occurrence_matrix(data):
    vectorizer = CountVectorizer(stop_words='english', binary=True)
    co_occurrence_matrix = vectorizer.fit_transform(data)
    return co_occurrence_matrix

# Function to perform concept modeling using NMF
def perform_concept_modeling(data, co_occurrence_matrix, num_topics):
    tfidf_vectorizer = TfidfTransformer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(co_occurrence_matrix)

    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_features = nmf_model.fit_transform(tfidf_matrix)

    return nmf_features

# Function to perform clustering with K-Means
def perform_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters

# Function to visualize and save data as images
def save_visualizations(data, true_labels, clusters):
    # Scatter plot of concept-based similarities with clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=clusters, palette='Dark2', legend='full')
    plt.title('Scatter Plot of Concept-Based Similarities with Clusters')
    plt.xlabel('Similarity Dimension 1')
    plt.ylabel('Similarity Dimension 2')
    scatter_plot_path = 'static/scatter_plot.png'
    plt.savefig(scatter_plot_path)

    # Pie chart of true label distribution
    plt.figure(figsize=(8, 6))
    true_labels_series = pd.Series(true_labels)
    true_labels_counts = true_labels_series.value_counts(normalize=True)
    plt.pie(true_labels_counts, labels=true_labels_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of True Labels')
    pie_chart_path = 'static/pie_chart.png'
    plt.savefig(pie_chart_path)

    # Bar graph of true label distribution
    plt.figure(figsize=(10, 6))
    true_labels_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('True Labels')
    plt.ylabel('Percentage')
    plt.title('Percentage Distribution of True Labels')
    plt.xticks(rotation=45)
    plt.tight_layout()
    bar_chart_path = 'static/bar_chart.png'
    plt.savefig(bar_chart_path)

    return scatter_plot_path, pie_chart_path, bar_chart_path

# Main function to orchestrate the entire process
def main(dataset_folder=DATASET_FOLDER):
    logging.info("Loading data...")
    articles, true_labels = load_data(dataset_folder)

    logging.info("Preprocessing text data...")
    processed_articles = preprocess_text_data(articles)

    logging.info("Calculating word co-occurrence matrix...")
    co_occurrence_matrix = calculate_co_occurrence_matrix(processed_articles)

    logging.info("Performing concept modeling with NMF...")
    num_topics = int(np.sqrt(len(processed_articles)))  # Determine number of topics automatically
    nmf_features = perform_concept_modeling(processed_articles, co_occurrence_matrix, num_topics)

    logging.info("Calculating concept-based similarities...")
    concept_similarities = cosine_similarity(nmf_features)

    logging.info("Performing clustering with K-Means on concept-based similarities...")
    num_clusters = min(5, len(articles))  # Limit clusters to a maximum of 5
    clusters = perform_clustering(concept_similarities, num_clusters)

    logging.info("Saving visualizations...")
    scatter_plot_path, pie_chart_path, bar_chart_path = save_visualizations(concept_similarities, true_labels, clusters)

    return scatter_plot_path, pie_chart_path, bar_chart_path

# Flask app code
app = Flask(__name__, static_folder='static')


@app.route('/', methods=['GET', 'POST'])
def index():
    scatter_plot_path, pie_chart_path, bar_chart_path = None, None, None
    if request.method == 'POST':
        print("Form submitted. Generating visualizations...")
        scatter_plot_path, pie_chart_path, bar_chart_path = main()  # Call the main function to generate and save the visualizations
        print("Visualizations generated:", scatter_plot_path, pie_chart_path, bar_chart_path)
    return render_template('index.html', scatter_plot_img=scatter_plot_path, pie_chart_img=pie_chart_path, bar_chart_img=bar_chart_path)

@app.route('/static/<path:filename>')
def serve_static(filename):
    root_dir = os.path.dirname(os.getcwd())
    return send_file(os.path.join(root_dir, 'static', filename))

if __name__ == '__main__':
    app.run(debug=True)
