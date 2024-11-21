# Databricks notebook source
# MAGIC %pip install -U openai 
# MAGIC %pip install dbtunnel[gradio] dspy-ai==2.5.29 pydantic
# MAGIC %pip install arize-otel==0.3.1 openai openinference-instrumentation-openai opentelemetry-sdk openinference-instrumentation-dspy opentelemetry-exporter-otlp
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_CONFIG

# COMMAND ----------

from auto_topic.domains import DomainConfigTable
from auto_topic.sentiment import get_analyzer, enable_arize_tracing, get_valid_responses_for_categories, get_when_to_use_category
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Enable tracing with arize UI
# MAGIC
# MAGIC If you do not need tracing at the llm call level comment out the following cell otherwise you need to make sure you are running the 00_TRACING notebook.

# COMMAND ----------

enable_arize_tracing()

# COMMAND ----------

dct = DomainConfigTable.from_table(spark, catalog=CATALOG, schema=SCHEMA, table=QUESTIONS_TABLE)

# COMMAND ----------

import dspy
import os


# configure these if you have the model in another workspace
# os.environ["DATABRICKS_API_BASE"] = ""
# os.environ["DATABRICKS_API_KEY"] = ""
# the model should always be "databricks/<endpoint name>"
language_model = dspy.LM(
    f'databricks/{MODEL_ID}',
     cache=False,
)

topics_df = pd.DataFrame([topic.to_kwargs() for topic in dct.topics])
valid_categories = get_valid_responses_for_categories(topics_df)
when = get_when_to_use_category(topics_df) 

# Look through the code to get more customization here
analyze = get_analyzer(topics_df, language_model)

# COMMAND ----------

def extract(feedback, value):
    print(f"Feedback: {feedback}, Rating: {value}")
    resp = analyze(feedback=feedback, rating=str(value))
    final_resp = {"category_breakdown": resp.breakdown.to_dict(), 
                  "category_selection": resp.category_selection,
                  "category_selection_rationale": resp.category_selection_rationale,
                  "all_categories": valid_categories}
    return final_resp

# COMMAND ----------

spark.table(f"{CATALOG}.{SCHEMA}.{REVIEWS_TABLE}").display()

# COMMAND ----------

import gradio as gr
demo = gr.Interface(
    fn=extract,
    inputs=["text",gr.Slider(minimum=1, maximum=5, step=1)],
    outputs=["json"],
)

from dbtunnel import dbtunnel
dbtunnel.gradio(demo).run()

# COMMAND ----------


