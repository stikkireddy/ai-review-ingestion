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

import lancedb
from pathlib import Path

# local
LOCAL_PATH = Path("/local_disk0/")
lancedb_path = str(Path(LOCAL_PATH) / "lance" / "reviews_db")
uri = lancedb_path
db = lancedb.connect(uri)

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

data = reviews_df.toPandas().to_dict(orient="records")
def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
chunk_size = 50
chunked_list = chunk_list(data, chunk_size)
final_list = []
for idx, chunk in enumerate(chunked_list):
    print(f"Processing {len(chunk)} records for chunk: {idx}")
    docs = [rec["review"] for rec in chunk]
    response = client.embeddings.create(
        model="databricks-gte-large-en",
        input=docs
    )
    for idx, rec in enumerate(chunk):
        rec["vector"] = response.data[idx].embedding
        final_list.append(rec)

# COMMAND ----------

final_list[0]

# COMMAND ----------

# /local
table = db.create_table(
    "reviews_lance_table",
    mode="overwrite",
    data=final_list,
)

# COMMAND ----------

table.create_fts_index("review", tokenizer_name="en_stem", replace=True)

# COMMAND ----------

def get_total_by(table, cond: dict[str, str] = None):
  cond = cond or {}
  where = " AND ".join(f"{k}='{v}'" for k, v in cond.items()) if cond != {} else None
  return table.count_rows(filter=where if cond != "" else None)

def text_query_by(table, search_string, cond: dict[str, str] = None, limit=100000, select=None):
  cond = cond or {}
  where_stmt = " AND ".join(f"{k}='{v}'" for k, v in cond.items()) if cond != "" else None
  if where_stmt:
      q = table.search(search_string).where(where_stmt, prefilter=True).limit(limit)
  else:
      q = table.search(search_string).limit(limit)
  if select:
      q = q.select(select)
      return q.to_list()
  else:
      return q.to_list()
    
def vector_query_by(table, search_string, cond: dict[str, str] = None, 
                    limit=100000, 
                    select=None,
                    threshold=0.6):
  cond = cond or {}
  where_stmt = " AND ".join(f"{k}='{v}'" for k, v in cond.items()) if cond != "" else None
  if where_stmt:
      q = table.search(get_embedding(search_string)).metric("cosine").where(where_stmt, prefilter=True).limit(limit)
  else:
      q = table.search(get_embedding(search_string)).metric("cosine").limit(limit)
  if select:
      selected_set = set(select)
      for key in cond.keys():
          selected_set.add(key)
      q = q.select(list(selected_set))
      return [i for i in q.to_list() if i["_distance"] < threshold]
  else:
      return [i for i in q.to_list() if 1-i["_distance"] < threshold]


# COMMAND ----------

table = db.open_table("reviews_lance_table")
table.search("phone").select(["review"]).to_list()

# COMMAND ----------

table.search(get_embedding("broken device does not work properly")).metric("cosine").select(["review"]).to_list()

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
        total_ct = get_total_by(table, cond=condition)
        txt_res = text_query_by(table, sentence.value, condition, select=["brand", "review", "review_id"])
        vec_res = vector_query_by(table, sentence.value, condition, select=["brand", "review", "review_id"], threshold=0.7)

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
                query_res_str = "\n".join([f'<li> {txt["review_id"]} - {txt["review"]} </li>' for txt in txt_res[:min(result_limit.value, len(txt_res))]])
                solara.HTML(tag="div", unsafe_innerHTML=f"""
                            <h3> Text Query Results (Max Results {result_limit.value}) </h3>
                            <ul>
                                {query_res_str}
                                </ul>
                            """)

            with solara.lab.Tab("Semantic"):
                query_res_str = "\n".join([f'<li> {txt["review_id"]} - {txt["review"]} </li>' for txt in vec_res[:min(result_limit.value, len(vec_res))]])
                solara.HTML(tag="div", unsafe_innerHTML=f"""
                            <h3> Semantic Similarity Query Results (Max Results {result_limit.value}) </h3>
                            <ul>
                                {query_res_str}
                                </ul>
                            """)
            

# The following line is required only when running the code in a Jupyter notebook:
Page()

# COMMAND ----------


