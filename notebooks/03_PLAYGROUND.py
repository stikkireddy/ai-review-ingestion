# Databricks notebook source
# MAGIC %pip install -U openai pandas dbtunnel[gradio] dspy-ai pydantic
# MAGIC %pip install arize-otel openai openinference-instrumentation-openai opentelemetry-sdk openinference-instrumentation-dspy opentelemetry-exporter-otlp
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_CONFIG

# COMMAND ----------

from auto_topic.domains import DomainConfigTable
from auto_topic.sentiment import get_analyzer, enable_arize_tracing, get_valid_responses_for_categories
import pandas as pd

# COMMAND ----------

enable_arize_tracing()

# COMMAND ----------

dct = DomainConfigTable.from_table(spark, catalog=CATALOG, schema=SCHEMA, table=QUESTIONS_TABLE)

# COMMAND ----------

import dspy
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
language_model = dspy.OpenAI(
    model='databricks-meta-llama-3-70b-instruct', # model='databricks-dbrx-instruct',
    max_tokens=500,
    temperature=0.1,
    api_key=TOKEN,
    api_base="https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/"
)
topics_df = pd.DataFrame([topic.to_row()["kwargs"] for topic in dct.topics])
analyze = get_analyzer(topics_df, language_model)

# COMMAND ----------

valid_categories = get_valid_responses_for_categories(topics_df) 
def extract(feedback, value):
    print(f"Feedback: {feedback}, Rating: {value}")
    resp = analyze(feedback=feedback, rating=str(value))
    final_resp = {"category_breakdown": resp.breakdown.to_dict(), 
                  "category_selection": resp.category_selection,
                  "category_selection_rationale": resp.category_selection_rationale,
                  "all_categories": valid_categories}
    return final_resp

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


