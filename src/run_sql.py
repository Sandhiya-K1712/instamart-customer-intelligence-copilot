
import sqlite3
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. CONNECT TO DATABASE
# --------------------------------------------------

DB_PATH = Path("data/instamart.db")
SQL_PATH = Path("sql/queries.sql")

connection = sqlite3.connect(DB_PATH)

# --------------------------------------------------
# 2. READ SQL QUERIES
# --------------------------------------------------

sql_text = SQL_PATH.read_text(encoding="utf-8")

# Split queries using semicolon
queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

# --------------------------------------------------
# 3. EXECUTE EACH QUERY USING PANDAS
# --------------------------------------------------

for index, query in enumerate(queries, start=1):

    print("\n" + "=" * 60)
    print(f"SQL QUERY Q{index}")
    print("=" * 60)

    result = pd.read_sql_query(query, connection)

    print(result.to_string(index=False))

    print("\nNumber of rows returned:", len(result))

connection.close()

print("\nSQL EXECUTION COMPLETED")
