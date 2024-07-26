# Databricks notebook source
# MAGIC %pip install tantivy==0.20.1 lancedb
# MAGIC %pip install -U openai
# MAGIC %pip install solara
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_CONFIG

# COMMAND ----------

from pyspark.sql import functions as F
reviews_df = spark.table(f"{CATALOG}.{SCHEMA}.{REVIEWS_TABLE}")
reviews_df.display()

# COMMAND ----------

from openai import OpenAI

TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
client = OpenAI(
    api_key=TOKEN,
    base_url=f"{WORKSPACE_URL}/serving-endpoints"
)

def get_embedding(inp: str):
  response = client.embeddings.create(
    model="databricks-gte-large-en",
    input=inp
  )
  return response.data[0].embedding

# COMMAND ----------

get_embedding("test the embeddings")

# COMMAND ----------

data = reviews_df.toPandas().to_dict(orient="records")

def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

chunk_size = 100
chunked_list = chunk_list(data, chunk_size)
final_data = []

for idx, chunk in enumerate(chunked_list):
    print(f"Processing {len(chunk)} records for chunk: {idx}")
    docs = [rec["review"] for rec in chunk]
    response = client.embeddings.create(
        model="databricks-gte-large-en",
        input=docs
    )
    for idx, rec in enumerate(chunk):
        rec["vector"] = response.data[idx].embedding
        final_data.append(rec)

# COMMAND ----------

final_data[0]

# COMMAND ----------

from auto_topic.index import Indexer
indexer = Indexer("reviews_index", f"{VOLUME_BASE_PATH}/indexes/reviews")
indexer.publish_index(final_data)

# COMMAND ----------

print("Total count of records", indexer.get_total_by())

# COMMAND ----------

import solara
import solara.lab

# Declare reactive variables at the top level. Components using these variables
# will be re-executed when their values change.
DEFAULT = "Query Text..."
sentence = solara.reactive(DEFAULT)
result_limit = solara.reactive(10)
brands =  ["All Brands"] + spark.sql(f"SELECT distinct brand from {CATALOG}.{SCHEMA}.{REVIEWS_TABLE}").toPandas()["brand"].tolist()
selected_brand = solara.reactive("All Brands")

import re

def highlight_text(input_string, query_string):
    """
    Replace items in the input_string with HTML <div> tags that highlight them in yellow.
    
    :param input_string: The string where replacements will be made.
    :param highlights: A list of strings to be highlighted.
    :return: A string with HTML <div> tags highlighting the items from the list.
    """
    highlights = query_string.split()
    for highlight in highlights:
        escaped_highlight = re.escape(highlight)
        # Replace each occurrence of the highlight text with a <div> tag containing the text
        input_string = re.sub(
            f'({escaped_highlight})', 
            r'<div style="background-color: yellow; display: inline;">\1</div>', 
            input_string, 
            flags=re.IGNORECASE
        )
    return input_string


@solara.component
def Page():
    # Calculate word_count within the component to ensure re-execution when reactive variables change.
    word_count = len(sentence.value.split())
    solara.Select("Select an brand", values=brands, value=selected_brand)
    solara.SliderInt("Result Limit", value=result_limit, min=5, max=50)
    solara.InputText(label="Your sentence", value=sentence, continuous_update=True)

    if selected_brand.value != "All Brands":
        condition = {"brand": selected_brand.value}
    else:
        condition = None
        
    if DEFAULT != sentence.value:
        total_ct = indexer.get_total_by(conditions=condition)
        txt_res = indexer.text_query_by(sentence.value, condition, select=["brand", "review", "review_id"])
        vec_res = indexer.vector_query_by(sentence.value, get_embedding, condition, select=["brand", "review", "review_id"], threshold=0.7)

        solara.HTML(tag="div", unsafe_innerHTML=f"""
                    <ul>
                        <li> Searched for: {sentence.value} </li>
                        <li> Total Records: {total_ct} </li>
                        <li> Total Txt Results: {len(txt_res)} </li>
                        <li> Total Semantic Results: {len(vec_res)} </li>
                    </ul>
                    """)
        
        with solara.lab.Tabs():
            with solara.lab.Tab("Text Query Results"):
                query_res_str = "\n".join([f'<li> {txt["review_id"]} - {highlight_text(txt["review"], sentence.value)} </li>' for txt in txt_res[:min(result_limit.value, len(txt_res))]])
                solara.HTML(tag="div", unsafe_innerHTML=f"""
                            <h3> Text Query Results (Max Results {result_limit.value}) </h3>
                            <ul>
                                {query_res_str}
                                </ul>
                            """)

            with solara.lab.Tab("Semantic"):
                query_res_str = "\n".join([f'<li> {txt["review_id"]} - {highlight_text(txt["review"], sentence.value)} </li>' for txt in vec_res[:min(result_limit.value, len(vec_res))]])
                solara.HTML(tag="div", unsafe_innerHTML=f"""
                            <h3> Semantic Similarity Query Results (Max Results {result_limit.value}) </h3>
                            <ul>
                                {query_res_str}
                                </ul>
                            """)
            

# The following line is required only when running the code in a Jupyter notebook:
Page()

# COMMAND ----------


