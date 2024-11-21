# Databricks notebook source
# MAGIC %pip install -U openai pandas dspy-ai==2.5.29 pydantic
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_CONFIG

# COMMAND ----------

from auto_topic.domains import DomainConfigTable
from auto_topic.sentiment import get_analyzer, enable_arize_tracing, get_valid_responses_for_categories, get_when_to_use_category, build_analysis_views
import pandas as pd

# COMMAND ----------

dct = DomainConfigTable.from_table(spark, catalog=CATALOG, schema=SCHEMA, table=QUESTIONS_TABLE)

# COMMAND ----------

TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
topics_df = pd.DataFrame([topic.to_kwargs() for topic in dct.topics])

# COMMAND ----------

topics_df

# COMMAND ----------

reviews = spark.table(f"{CATALOG}.{SCHEMA}.{REVIEWS_TABLE}")
reviews = reviews.limit(10)
reviews.display()

# COMMAND ----------

import json
import pandas as pd
from multiprocessing.pool import ThreadPool
from typing import Any
import dspy

def analyze_reviews(df: pd.DataFrame, topics_df: pd.DataFrame) -> pd.DataFrame:
    import os
    # configure these if you have the model in another workspace
    # os.environ["DATABRICKS_API_BASE"] = ""
    # os.environ["DATABRICKS_API_KEY"] = ""
    # the model should always be "databricks/<endpoint name>"
    # add other things like top-k or temperature
    language_model = dspy.LM(
        f'databricks/{MODEL_ID}',
        cache=False,
        max_tokens=4000,
    )
    extract = get_analyzer(topics_df, language_model)
    
    # Function to process individual review and rating
    def process_feedback(feedback: str, rating: Any) -> str:
        resp = extract(feedback=feedback, rating=str(rating))
        return json.dumps({
            "category_breakdown": resp.breakdown.to_dict(),
            "category_selection": resp.category_selection,
            "category_selection_rationale": resp.category_selection_rationale,
            "all_categories": get_valid_responses_for_categories(topics_df)
        })
    
    # Extract reviews and ratings columns
    feedbacks = df["review"].tolist()
    ratings = df["rating"].tolist()

    # Create a thread pool with 10 workers and process in parallel
    with ThreadPool(100) as pool:
        results = pool.starmap(process_feedback, zip(feedbacks, ratings))

    # Assign results to a new column in the DataFrame
    df["analysis"] = results
    return df

# spark.createDataFrame(analyze_reviews(reviews.toPandas(), topics_df)).display()

# COMMAND ----------

# reviews = reviews.withColumn("analysis", extract_domain_details("review", "rating"))
# reviews.display()

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.{TARGET_TABLE}")

# COMMAND ----------

#  REMOVE LIMIT 10
if spark.catalog.tableExists(f"{CATALOG}.{SCHEMA}.{TARGET_TABLE}") is False:
    print(f"Creating Target Table: {CATALOG}.{SCHEMA}.{TARGET_TABLE}")
    spark.sql(f"""
              SELECT *, cast(null as string) as analysis FROM {CATALOG}.{SCHEMA}.{REVIEWS_TABLE}
               LIMIT 10 
              """).write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.{TARGET_TABLE}")
else:
    print(f"Table: {CATALOG}.{SCHEMA}.{TARGET_TABLE} already exists!")

# COMMAND ----------

# seed reviews
display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.{TARGET_TABLE};"))

# COMMAND ----------

from delta.tables import DeltaTable

target_table = DeltaTable.forName(spark, f"{CATALOG}.{SCHEMA}.{TARGET_TABLE}")

records = spark.sql(f"SELECT *, cast(null as string) as analysis FROM {CATALOG}.{SCHEMA}.{REVIEWS_TABLE}")

target_table.alias("target").merge(
    source=records.alias("source"),
    condition="target.review_id = source.review_id",
).whenNotMatchedInsertAll().execute()

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {CATALOG}.{SCHEMA}.{TARGET_TABLE};"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # ETL Logic
# MAGIC
# MAGIC 1. Identify unanalyzed columns (where analysis is null)
# MAGIC 2. Then chunk the amount of rows to process via BATCH_ETL_BATCH_SIZE and then commit the transaction via a merge.
# MAGIC 3. Repeat till the analysis columns are not null
# MAGIC
# MAGIC
# MAGIC If there are 1000 null records and BATCH_ETL_BATCH_SIZE=500 you will have 2 transcations made to the table. 

# COMMAND ----------

from delta.tables import DeltaTable

def get_unanalyzed_records(spark, batch_size = None):
    if batch_size is None:
        return spark.table(f"{CATALOG}.{SCHEMA}.{TARGET_TABLE}").where("analysis is null")
    else:
        return spark.table(f"{CATALOG}.{SCHEMA}.{TARGET_TABLE}").where("analysis is null").limit(batch_size)

unanalyzed_records = get_unanalyzed_records(spark, 100)
unanalyzed_records_ct = unanalyzed_records.count()
unanalyzed_records_ct
print(unanalyzed_records_ct)
batch_ct = 1
while unanalyzed_records_ct > 0:
    print(f"Analyzing {unanalyzed_records_ct} records...; batch number {batch_ct}")
    unanalyzed_records_pdf = unanalyzed_records.toPandas()
    print(unanalyzed_records_pdf.shape)
    analyized_pdf = analyze_reviews(unanalyzed_records_pdf, topics_df)
    analyzed_records = spark.createDataFrame(analyized_pdf)
    target_table = DeltaTable.forName(spark, f"{CATALOG}.{SCHEMA}.{TARGET_TABLE}")
    target_table.alias("target").merge(
        source=analyzed_records.alias("source"),
        condition="target.review_id = source.review_id",
    ).whenMatchedUpdateAll().execute()

    # fetch new batch
    unanalyzed_records = get_unanalyzed_records(spark, BATCH_ETL_BATCH_SIZE)
    unanalyzed_records_ct = unanalyzed_records.count()
    batch_ct += 1




# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Analysis Views
# MAGIC
# MAGIC Build views to analyze data along with comments. These tables can be used for genie data room

# COMMAND ----------

build_analysis_views(
    spark=spark,
    catalog=CATALOG,
    schema=SCHEMA,
    analysis_table=TARGET_TABLE, 
    primary_key_col_name="review_id", 
    domain_config_table=dct
)

# COMMAND ----------

spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Identify Processed Data
# MAGIC
# MAGIC Select data that has been analyzed already

# COMMAND ----------

display(spark.sql(f"""
SELECT * FROM {CATALOG}.{SCHEMA}.{TARGET_TABLE}
WHERE analysis is not null;
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Identify errors
# MAGIC
# MAGIC Identify failed analysis tables by looking for `WHERE analysis:category_breakdown:error is not null`

# COMMAND ----------

# Failures in the analysis column
display(spark.sql(f"""
SELECT * FROM {CATALOG}.{SCHEMA}.{TARGET_TABLE}
WHERE analysis:category_breakdown:error is not null;
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Reprocess Data
# MAGIC
# MAGIC Set the analysis columns to null to have this notebook reprocess them.

# COMMAND ----------

# RESET EVERYTHING TO REPROCESS
# spark.sql(f"""
# UPDATE {CATALOG}.{SCHEMA}.{TARGET_TABLE} 
# SET analysis = null 
# WHERE analysis:category_breakdown:error is not null;
# """)
