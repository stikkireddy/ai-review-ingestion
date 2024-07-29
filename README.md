# AI Review Ingestion

## Purpose

This repository provides a set of tools for generating topics from sets of questions, predetermined topics,
or zero-shot using various different techniques such as LLMs or BERT-based models.

## Technologies

This repository uses the following technologies:

1. DSPy: Used for prompt management and in future for prompt optimization and bootstrapping multi shot prompts
2. Arize: Temporarily used for monitoring and debugging dspy till mlflow tracing support is added
3. LanceDB: Is temporarily used for the search tool along with UC volumes to be able to FTS with unlimited result set to
   do aggs
4. Llama 3.1 70b: Used for topic generation and sentiment analysis using dspy as the prompt management tool
5. All data is stored in delta tables

## High-level Design

The repository is organized into several modules:

* `auto_topic.index`: Provides functionality for indexing and searching text data.
* `auto_topic.sentiment`: Analyzes text data to extract sentiment and other relevant information.
* `auto_topic.domains`: Defines domains and questions to be answered by the LLM.
* `auto_topic.source`: Specifies the source of the text data.
* `auto_topic.extract`: Extracts relevant information from the text data.

## How to use

1. Install the required packages by running `pip install -r requirements.txt`.
2. Set up your data by creating a table with the required columns (e.g., `review_id`, `rating`, `review`).
3. Configure the `00_CONFIG` notebook with your specific settings (e.g., `CATALOG`, `SCHEMA`, `REVIEWS_TABLE`).
4. Run the `01_SETUP_DATA` notebook to set up your data.
5. Run the `05_SEARCH_TOOL` notebook to create an index and search for text data.
6. Run the `04_BATCH_ETL` notebook to extract relevant information from the text data.

## Notebook Explanations

### 00_CONFIG

Configures the settings for the repository, including the catalog, schema, and table names.

### 01_SETUP_DATA

Sets up the data by creating a table with the required columns.

### 02_SETUP_DOMAINS

Defines domains and questions to be answered by the LLM using the `auto_topic.domains` module.

### 03_PLAYGROUND

Provides a playground for testing and experimenting with the repository.

### 04_BATCH_ETL

Extracts relevant information from the text data using the `auto_topic.extract` module.

### 05_SEARCH_TOOL

Creates an index and searches for text data using the `auto_topic.index` module.

## Optional Notebooks Explained

### 00_TRACING (Optional)

Enables tracing with Arize UI for monitoring and debugging purposes.

### 01_SETUP_TOPIC_ANALYSIS (Optional but Recommended)

Analyzes the text data to extract topics and sentiment using the `auto_topic.sentiment` module.

Note: This notebook is optional but recommended for gaining a deeper understanding of the text data.

### Example Use Cases

* Analyzing customer reviews to extract sentiment and topics.
* Identifying defects and issues with products.
* Extracting relevant information from text data for decision-making purposes.

### Troubleshooting

* Check the `00_CONFIG` notebook for correct settings.
* Verify that the required packages are installed.
* Consult the documentation for each module for specific troubleshooting tips.

## Disclaimer

This is a "what you see is what you get" set of notebooks, and it is your responsibility to use them to go to
production. The notebooks are provided as examples, and you should modify them to suit your specific needs.
