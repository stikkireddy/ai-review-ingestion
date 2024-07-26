from pathlib import Path
import shutil
import lancedb
from typing import Callable

def copy_recursive(source: Path, target: Path):
    """
    Recursively copy all contents from source directory to target directory.

    Parameters:
    - source: The source directory path (Path object).
    - target: The target directory path (Path object).
    """
    if not source.is_dir():
        raise ValueError(f"Source path '{source}' is not a directory.")
    
    # Create target directory if it doesn't exist
    target.mkdir(parents=True, exist_ok=True)
    
    for item in source.iterdir():
        # Determine the corresponding target path
        target_item = target / item.name
        
        if item.is_dir():
            # Recursively copy subdirectories
            copy_recursive(item, target_item)
        else:
            # Copy files
            shutil.copy2(item, target_item)
            print(f"Copied file '{item}' to '{target_item}'")


class Indexer:

    def __init__(self, 
                 index_name,
                 target_location, 
                 staging_location="/local_disk0/lancedb/staging",
                 tmp_location="/local_disk0/lancedb/tmp"):
        self.index_name = index_name
        self.target_location = Path(target_location)
        self.staging_location: Path = Path(staging_location)
        self.tmp_location : Path = Path(tmp_location)
        self.table = None
    
    def _copy_from_staging_recursive(self):
        print(f"Copying data from {self.staging_location} to {self.target_location}")
        self.target_location.mkdir(parents=True, exist_ok=True)
        copy_recursive(self.staging_location, self.target_location)

    def _clean_target_location(self):
        if self.target_location.exists():
            shutil.rmtree(str(self.target_location))

    def _clean_tmp_location(self):
        if self.tmp_location.exists():
            shutil.rmtree(str(self.tmp_location))

    def _copy_to_tmp_location(self):
        print(f"Copying data from {self.target_location} to {self.tmp_location}")
        self.tmp_location.mkdir(parents=True, exist_ok=True)
        copy_recursive(self.target_location, self.tmp_location)

    def publish_index(self,
                     data: list[dict], 
                     text_column_name="review", 
                     tokenizer_name="en_stem",
                     mode="overwrite"):
        print(f"Creating index {self.index_name} in {self.staging_location}")
        db = lancedb.connect(str(self.staging_location))
        table = db.create_table(
            self.index_name,
            mode="overwrite",
            data=data,
        )
        table.compact_files()
        table.create_fts_index(text_column_name, tokenizer_name=tokenizer_name, replace=True)
        table.compact_files()
        table.cleanup_old_versions()
        self._clean_target_location()
        self._copy_from_staging_recursive()
        return table
      
    def load_remote_table(self):
        self._clean_tmp_location()
        self._copy_to_tmp_location()
        db = lancedb.connect(str(self.tmp_location))
        self.table = db.open_table(self.index_name)

    def ensure_table(self):
        if self.table is None:
            self.load_remote_table()

    def get_total_by(self, conditions: dict[str, str] = None):
        self.ensure_table()
        conditions = conditions or {}
        where = " AND ".join(f"{k}='{v}'" for k, v in conditions.items()) if conditions != {} else None
        return self.table.count_rows(filter=where if conditions != "" else None)

    def text_query_by(self, search_string, conditions: dict[str, str] = None, limit=100000, select=None):
        self.ensure_table()
        conditions = conditions or {}
        where_stmt = " AND ".join(f"{k}='{v}'" for k, v in conditions.items()) if conditions != "" else None
        if where_stmt:
            q = self.table.search(search_string).where(where_stmt, prefilter=True).limit(limit)
        else:
            q = self.table.search(search_string).limit(limit)
        if select:
            q = q.select(select)
            return q.to_list()
        else:
            return q.to_list()
    
    def vector_query_by(self, 
                        search_string, 
                        embedding_func: Callable,
                        conditions: dict[str, str] = None, 
                        limit=100000, 
                        select=None,
                        threshold=0.6):
        self.ensure_table()
        get_embedding = embedding_func
        conditions = conditions or {}
        where_stmt = " AND ".join(f"{k}='{v}'" for k, v in conditions.items()) if conditions != "" else None
        if where_stmt:
            q = self.table.search(get_embedding(search_string)).metric("cosine").where(where_stmt, prefilter=True).limit(limit)
        else:
            q = self.table.search(get_embedding(search_string)).metric("cosine").limit(limit)
        if select:
            selected_set = set(select)
            for key in conditions.keys():
                selected_set.add(key)
            q = q.select(list(selected_set))
            return [i for i in q.to_list() if i["_distance"] < threshold]
        else:
            return [i for i in q.to_list() if 1-i["_distance"] < threshold]

        return db.open_table(self.index_name)



