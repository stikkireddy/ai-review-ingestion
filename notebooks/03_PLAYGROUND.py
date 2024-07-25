# Databricks notebook source
# MAGIC %pip install -U openai pandas dbtunnel[gradio] dspy-ai pydantic
# MAGIC %pip install arize-otel openai openinference-instrumentation-openai opentelemetry-sdk openinference-instrumentation-dspy opentelemetry-exporter-otlp
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_CONFIG

# COMMAND ----------

from auto_topic.domains import DomainConfigTable
from auto_topic.sentiment import get_analyzer, enable_arize_tracing, get_valid_responses_for_categories, get_when_to_use_category
import pandas as pd

# COMMAND ----------

enable_arize_tracing()

# COMMAND ----------

dct = DomainConfigTable.from_table(spark, catalog=CATALOG, schema=SCHEMA, table=QUESTIONS_TABLE)

# COMMAND ----------

import dspy
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
language_model = dspy.OpenAI(
    model=MODEL_ID, # model='databricks-dbrx-instruct',
    max_tokens=500,
    temperature=0.1,
    api_key=TOKEN,
    api_base=f"{WORKSPACE_URL}/serving-endpoints/"
)

topics_df = pd.DataFrame([topic.to_kwargs() for topic in dct.topics])
valid_categories = get_valid_responses_for_categories(topics_df)
when = get_when_to_use_category(topics_df) 

class IdentifyCategories(dspy.Signature):
    """You are a customer feedback analyst. Your work at online retailer and oversee customer feedback across all channels, such as website, social media, in store reviews, customer reviews via email and your goal is to extract meaning from the customer reviews. Make sure when asked for json you start with { and end with }"""

    feedback = dspy.InputField(desc="a review from a customer. this can be a positive or negative review")
    categories = dspy.OutputField(desc="the categories of the feedback. one of: " + ", ".join(valid_categories) + f" and here is when to use each category {str(when)}. Respond with catchalldetails if not related to any of the categories. You can pick multiple categories and respond with a comma separated list.")
    rationale = dspy.OutputField(desc="rationale for picking the categories explain step by step")

class Categorize(dspy.Signature):
    """You are a customer feedback analyst. Your work at online retailer and oversee customer feedback across all channels, such as website, social media, in store reviews, customer reviews via email and your goal is to extract meaning from the customer reviews. Make sure when asked for json you start with { and end with }"""

    feedback = dspy.InputField(desc="a review from a customer. this can be a positive or negative review")
    rating = dspy.InputField(desc="use this for the connotation of the feedback. the scale ranges from 1 - 5. 1 being horrible 5 being great")

analyze = get_analyzer(topics_df, language_model, IdentifyCategories, Categorize)

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

import gradio as gr
demo = gr.Interface(
    fn=extract,
    inputs=["text",gr.Slider(minimum=1, maximum=5, step=1)],
    outputs=["json"],
)

from dbtunnel import dbtunnel
dbtunnel.gradio(demo).run()

# COMMAND ----------


