# Databricks notebook source
# MAGIC %pip install bertopic
# MAGIC %pip install -U openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_CONFIG

# COMMAND ----------

reviews_df = spark.table(f"{CATALOG}.{SCHEMA}.{REVIEWS_TABLE}").toPandas()

# COMMAND ----------

all_reviews = reviews_df["review"].tolist()

# COMMAND ----------

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI
import openai

# You can bootstrap with basic topics that you may want to pull
# Try picking relevant topics that you want your data clustered to
# you can also split up the parent review into sentence chunks and associate sentences to topics to be able to map a single review to one of these topics
zeroshot_topic_list = ["Good Movie", "Bad Movie", "Good Game", "Bad Game", "Other", "Defect Issues Not Working", "Gifts"]

# Reference: https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html#example
# if you are doing this with a lot of data you may want to use a gpu to use the embedding model otherwise 
# this can run for a while

TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
client = openai.OpenAI(base_url=f"{WORKSPACE_URL}/serving-endpoints/", api_key=TOKEN)
openai_generator = OpenAI(client, model=MODEL_ID)

ai_representation = [MaximalMarginalRelevance(diversity=0.4), openai_generator]

representations = {
    "AiBased": ai_representation,
    "KeyBERT": KeyBERTInspired()
}

topic_model = BERTopic(
    embedding_model="thenlper/gte-large", # use thenlper/gte-large or thenlper/gte-small; use small if you want this to run fast
    min_topic_size=15,
    zeroshot_topic_list=zeroshot_topic_list,
    zeroshot_min_similarity=.85,
    representation_model=representations # see if there is a better representation model
)
topics, _ = topic_model.fit_transform(all_reviews)

# COMMAND ----------

spark.createDataFrame(topic_model.get_topic_info()).display()

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


