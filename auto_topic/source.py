from typing import Union, Type, List

from pydantic import BaseModel, Field


class ExtractorTextSourceColumn(BaseModel):
    name: str
    data_type: Type[Union[str, int, float, bool]]
    description: str
    is_extractor_input: bool = False
    is_target_column: bool = True


SPARK_SQL_TYPE_MAP = {
    str: "STRING",
    int: "INT",
    float: "DOUBLE",
    bool: "BOOLEAN"
}

class ExtractorTextSourceTable(BaseModel):
    catalog: str
    schema: str
    table: str
    target_table_name: str
    target_column_name: str
    columns: List[ExtractorTextSourceColumn] = Field(default_factory=list)

    def with_column(self, name: str, data_type: Type[Union[str, int, float, bool]], description: str = "", is_extractor_input: bool = False, is_target_column: bool = True):
        self.columns.append(ExtractorTextSourceColumn(name=name, data_type=data_type, description=description,
                                                      is_extractor_input=is_extractor_input,
                                                      is_target_column=is_target_column))
        return self

    def create_table_statement(self):
        return f"CREATE TABLE {self.catalog}.{self.schema}.{self.table} ({', '.join([f'{col.name} {SPARK_SQL_TYPE_MAP[col.data_type]}' for col in self.columns])});"

    def select_target_columns(self, spark: "SparkSession"):
        columns_to_select = ", ".join([col.name for col in self.columns if col.is_target_column])
        return spark.sql(f"SELECT {columns_to_select} FROM {self.catalog}.{self.schema}.{self.table}")

    def create_target_table_statement(self):
        return (f"CREATE TABLE {self.catalog}.{self.schema}.{self.target_table_name} "
                f"({', '.join([f'{col.name} {SPARK_SQL_TYPE_MAP[col.data_type]}' for col in self.columns if col.is_extractor_input])}, {self.target_column_name} STRING);")

