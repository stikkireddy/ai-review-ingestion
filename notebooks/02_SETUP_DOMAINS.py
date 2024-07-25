# Databricks notebook source
# MAGIC %run ./00_CONFIG

# COMMAND ----------

from auto_topic.domains import DomainConfigTable, Domain

# COMMAND ----------

dct = DomainConfigTable(catalog=CATALOG, schema=SCHEMA, table=QUESTIONS_TABLE)

# COMMAND ----------

dct.with_topic(
  topic=Domain(
    topic="product",
    when="generally when the user is referering to the product they have purchased",
    details="<summary of the review regarding the product>"
  ).with_additional_info(
    item_name="can_be_used_as_gift",
    item_description="the product was used as a gift or will be used as a gift <yes/no/unsure>"
  ).with_additional_info(
    item_name="as_advertized",
    item_description="the product behaved as advertized whether it be description or other reviews <yes/no/unsure>"
  )
)

dct.with_topic(
  topic=Domain(
    topic="defects",
    when="generally when the user is referering to the product they have purchased",
    details="<summary of the review regarding the product>",
  ).with_additional_info(
    item_name="defect_level",
    item_description="the level of defectiveness to the product use your best judgement <small/medium/large>"
  ).with_additional_info(
    item_name="misinformation",
    item_description="the product does not work as described <yes/no>"
  )
)


dct.with_topic(
  topic=Domain(
    topic="delivery",
    when="generally when the user is referering to the product they have purchased",
    details="<summary of the review regarding the product>",
  ).with_additional_info(
    item_name="defect_level",
    item_description="the level of defectiveness to the product use your best judgement <small/medium/large>"
  ).with_additional_info(
    item_name="misinformation",
    item_description="the product does not work as described <yes/no>"
  )
)

dct.with_topic(
  topic=Domain(
    topic="movies",
    when="generally when the user is referering to the movies they have purchased",
    details="<summary of the review regarding the product>",
  ).with_additional_info(
    item_name="enjoyment",
    item_description="the level of enjoyment to the moview use your best judgement <1-5>"
  ).with_additional_info(
    item_name="genre",
    item_description="the genre of the movie"
  ).with_additional_info(
    item_name="audio_issues",
    item_description="were there audio quality issues <yes/no>"
  ).with_additional_info(
    item_name="audio_issues",
    item_description="were there audio quality issues <yes/no>"
  )
)

# COMMAND ----------

dct.setup(spark)

# COMMAND ----------

spark.table(f"{dct.catalog}.{dct.schema}.{dct.table}").display()

# COMMAND ----------


