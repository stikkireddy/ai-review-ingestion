# Databricks notebook source
# MAGIC %pip install ../
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "main"
SCHEMA = "sri_winedb"
REVIEWS_TABLE = "reviews"
QUESTIONS_TABLE = "questions"
TARGET_TABLE = "reviews_predictions"
WORKSPACE_URL = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}'
MODEL_ID = "databricks-meta-llama-3-1-70b-instruct"
BATCH_ETL_BATCH_SIZE = 10 # make this 1000 or 10000 what ever batch size you can afford to fail on
VOLUME_BASE_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/sri_wine_volume" # modify this if you want to change the catalog or schema, volumes are at schema level
print("Variable: CATALOG = {}".format(CATALOG))
print("Variable: SCHEMA = {}".format(SCHEMA))
print("Variable: QUESTIONS_TABLE = {}".format(QUESTIONS_TABLE))
print("Variable: REVIEWS_TABLE = {}".format(REVIEWS_TABLE))
print("Variable: TARGET_TABLE = {}".format(TARGET_TABLE))
print("Variable: WORKSPACE_URL = {}".format(WORKSPACE_URL))
print("Variable: MODEL_ID = {}".format(MODEL_ID))
print("Variable: BATCH_ETL_BATCH_SIZE = {}".format(BATCH_ETL_BATCH_SIZE))
print("Variable: VOLUME_BASE_PATH = {}".format(VOLUME_BASE_PATH))

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------


