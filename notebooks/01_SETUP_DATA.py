# Databricks notebook source
# MAGIC %run ./00_CONFIG

# COMMAND ----------

spark.sql(f"""
          CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.{REVIEWS_TABLE} AS
          SELECT row_number() OVER (ORDER BY asin, user) as review_id, * 
          FROM parquet.`dbfs:/databricks-datasets/amazon/test4K/`
          """)


# COMMAND ----------

spark.table(f"{CATALOG}.{SCHEMA}.{REVIEWS_TABLE}").display()

# COMMAND ----------


