# Databricks notebook source
# MAGIC %pip install tantivy==0.20.1 lancedb
# MAGIC %pip install -U openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_CONFIG

# COMMAND ----------

from pyspark.sql import functions as F
reviews_df = spark.table(f"{CATALOG}.{SCHEMA}.{REVIEWS_TABLE}").withColumn("text", F.col("review"))
reviews_df.display()

# COMMAND ----------

import lancedb
from pathlib import Path

lancedb_path = str(Path(VOLUME_BASE_PATH) / "lance" / "reviews_db")
uri = lancedb_path
db = lancedb.connect(uri)

# COMMAND ----------

from openai import OpenAI

TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

client = OpenAI(
    api_key=TOKEN,
    base_url=f"{WORKSPACE_URL}/serving-endpoints"
)

response = client.embeddings.create(
  model="databricks-gte-large-en",
  input=["what is databricks"]
)
print(response.data)

# COMMAND ----------

data = reviews_df.toPandas()[["review_id", "text"]].to_dict(orient="records")
def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
chunk_size = 50
chunked_list = chunk_list(data, chunk_size)
final_list = []
for idx, chunk in enumerate(chunked_list):
    print(f"Processing {len(chunk)} records for chunk: {idx}")
    docs = [rec["text"] for rec in chunk]
    response = client.embeddings.create(
        model="databricks-gte-large-en",
        input=docs
    )
    for idx, rec in enumerate(chunk):
        rec["embedding"] = response.data[idx].embedding
        final_list.append(rec)

# COMMAND ----------

final_list

# COMMAND ----------

table = db.create_table(
    "reviews_lance_table",
    mode="overwrite",
    data=data,
)

# COMMAND ----------

table.create_fts_index("text", tokenizer_name="en_stem", replace=True)

# COMMAND ----------

len(table.search("issue").limit(100000).to_list())

# COMMAND ----------


