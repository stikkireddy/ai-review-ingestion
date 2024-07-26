# Databricks notebook source
# MAGIC %pip install bertopic
# MAGIC %pip install --upgrade openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_CONFIG

# COMMAND ----------

reviews_df = spark.table(f"{CATALOG}.{SCHEMA}.{REVIEWS_TABLE}").toPandas()

# COMMAND ----------

all_reviews = reviews_df["review"].tolist()

# COMMAND ----------

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

zeroshot_topic_list = ["Good Movie", "Bad Movie", "Good Game", "Bad Game", "Other", "Defect Issues Not Working", "Gifts"]

# Reference: https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html#example
topic_model = BERTopic(
    embedding_model="thenlper/gte-large", 
    min_topic_size=15,
    zeroshot_topic_list=zeroshot_topic_list,
    zeroshot_min_similarity=.75,
    representation_model=KeyBERTInspired()
)
topics, _ = topic_model.fit_transform(all_reviews)

# COMMAND ----------

topic_model.get_topic_info()

# COMMAND ----------

topic_distr, _ = topic_model.approximate_distribution(["the movie was great but was having some audio glitches along the way which put off the experience"])
topic_distr

# COMMAND ----------

topic_model.visualize_distribution(topic_distr[0])

# COMMAND ----------

new_document_topic, topic_probabilities = topic_model.transform(["the movie was great but was having some audio glitches along the way which put off the experience"])
# Get the topic ID assigned to the new document
topic_id = new_document_topic[0]
# Get the topic words for the assigned topic
topic_words = topic_model.get_topic(topic_id)
topic_string = ", ".join([word for word, _ in topic_words])
print(f"The new document is related to Topic {topic_id}: {topic_string}")
print(topic_probabilities)

# COMMAND ----------


