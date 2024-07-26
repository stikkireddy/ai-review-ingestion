# Databricks notebook source
# MAGIC %run ./00_CONFIG

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Set up your data to be processed
# MAGIC
# MAGIC At bare minimum you need three columns:
# MAGIC
# MAGIC 1. review_id column (preferably named review_id) or you can modify subsequent notebooks
# MAGIC 2. rating column (numerical and preferably named rating) that ranges from some value
# MAGIC 3. review column (preferably named review) that is text in english
# MAGIC
# MAGIC You can have additional columns if that is important but not relevant for the rest of the notebooks.

# COMMAND ----------

spark.sql(f"""
          CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.{REVIEWS_TABLE} AS
          SELECT row_number() OVER (ORDER BY asin, user) as review_id, * 
          FROM parquet.`dbfs:/databricks-datasets/amazon/test4K/`
          """)


# COMMAND ----------

spark.table(f"{CATALOG}.{SCHEMA}.{REVIEWS_TABLE}").display()

# COMMAND ----------


