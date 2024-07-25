# Databricks notebook source
# MAGIC %pip install -U openai pandas dbtunnel[gradio] dspy-ai pydantic
# MAGIC # %pip install arize-otel openai openinference-instrumentation-openai opentelemetry-sdk openinference-instrumentation-dspy opentelemetry-exporter-otlp
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_CONFIG

# COMMAND ----------

from auto_topic.domains import DomainConfigTable
from auto_topic.sentiment import get_analyzer, enable_arize_tracing, get_valid_responses_for_categories
import pandas as pd

# COMMAND ----------

dct = DomainConfigTable.from_table(spark, catalog=CATALOG, schema=SCHEMA, table=QUESTIONS_TABLE)

# COMMAND ----------

import dspy
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
topics_df = pd.DataFrame([topic.to_kwargs() for topic in dct.topics])

# COMMAND ----------

topics_df

# COMMAND ----------

from pyspark.sql import functions as F
import json
import dspy

from typing import Iterator, Tuple

@F.pandas_udf("string")
def extract_domain_details(feedback_and_ratings: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:
    # Do some expensive initialization with a state
    language_model = dspy.OpenAI(
        model='databricks-meta-llama-3-70b-instruct', # model='databricks-dbrx-instruct',
        max_tokens=500,
        temperature=0.1,
        api_key=TOKEN,
        api_base="https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/"
    )
    extract = get_analyzer(topics_df, language_model)
    for feedback_arr, rating_arr in feedback_and_ratings:
        # Use that state for whole iterator.
        feedbacks, ratings = feedback_arr.tolist(), rating_arr.tolist()
        results = []
        for feedback, rating in zip(feedbacks, ratings):
            resp = extract(feedback=feedback, rating=str(rating))
            final_resp = json.dumps({"category_breakdown": resp.breakdown.to_dict(), 
                    "category_selection": resp.category_selection,
                    "category_selection_rationale": resp.category_selection_rationale,
                    "all_categories": get_valid_responses_for_categories(topics_df)})
            results.append(final_resp)

        yield pd.Series(results)
  


# COMMAND ----------

reviews = spark.table(f"{CATALOG}.{SCHEMA}.{REVIEWS_TABLE}")
reviews.display()
reviews = reviews.limit(1)

# COMMAND ----------

reviews = reviews.withColumn("analysis", extract_domain_details("review", "rating"))
reviews.display()

# COMMAND ----------


