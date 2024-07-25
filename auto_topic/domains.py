import json
from typing import Optional, List

from pydantic import BaseModel, Field, validator


class DomainExtras(BaseModel):
    item_name: str
    item_description: str


class Domain(BaseModel):
    topic: str
    when: str
    details: str
    sentiment: str = "<positive/negative/mixed/neutral>"
    keywords: str = "<keyword1>,<keyword2>,..."
    ai_rating: Optional[str] = "generate a rating between <1-5/null> based on feedback with regards to {topic}"
    rationale: str = "explain your thoughts step by step on why you arrived at the conclusion with regards to {topic}"
    extras: List[DomainExtras] = Field(default_factory=list)

    @validator("ai_rating", pre=True, always=True)
    @classmethod
    def ai_rating(cls, v, values):
        return f"generate a rating between <1-5/null> based on feedback with regards to {values['topic']}"

    @validator("rationale", pre=True, always=True)
    @classmethod
    def rationale(cls, v, values):
        return f"explain your thoughts step by step on why you arrived at the conclusion with regards to {values['topic']}"

    def with_additional_info(self, item_name: str, item_description: str):
        self.extras.append(DomainExtras(item_name=item_name, item_description=item_description))
        return self
    
    @staticmethod
    def serialization_key():
        return "serialized"
    
    def to_kwargs(self):
        current_data = self.dict()
        current_data.pop("extras")
        for v in self.extras:
            current_data[v.item_name] = v.item_description.format(topic=self.topic)
        current_data["raw"] = json.dumps(current_data)
        return current_data

    def to_row(self):
        return {
            "topic": self.topic,
            self.serialization_key(): self.dict(),
        }


class DomainConfigTable(BaseModel):
    catalog: str
    schema: str
    table: str
    topics: List[Domain] = Field(default_factory=list)

    def with_topic(self, topic: Domain):
        self.topics.append(topic)
        return self

    def create_table_statement(self):
        return f"CREATE TABLE IF NOT EXISTS {self.catalog}.{self.schema}.{self.table} (topic STRING, raw STRING);"

    def insert_statements(self) -> list[str]:
        return [f"INSERT INTO {self.catalog}.{self.schema}.{self.table} VALUES ('{topic.topic}', '{json.dumps(topic.to_row())}');"
                for topic in self.topics]

    def truncate_statement(self):
        return f"TRUNCATE TABLE {self.catalog}.{self.schema}.{self.table};"

    @classmethod
    def from_table(cls, spark: "SparkSession", catalog: str, schema: str, table: str):
        rows = spark.table(f"{catalog}.{schema}.{table}").collect()
        topics = []
        for row in rows:
            topic = Domain(**json.loads(row.raw)[Domain.serialization_key()])
            topics.append(topic)
        return cls(catalog=catalog, schema=schema, table=table, topics=topics)

    def setup(self, spark):
        spark.sql(self.create_table_statement())
        spark.sql(self.truncate_statement())
        for insert_statement in self.insert_statements():
            spark.sql(insert_statement)
        return self
