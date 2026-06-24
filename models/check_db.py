import duckdb
from pathlib import Path

db_path = Path("./data/pipeline.duckdb")
if not db_path.exists():
    print("Database not found!")
else:
    conn = duckdb.connect(str(db_path))
    tables = conn.execute("SHOW TABLES").fetchall()
    print("Tables in DuckDB:")
    for t in tables:
        name = t[0]
        count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  - {name}: {count} rows")
        
        # Sample columns
        cols = conn.execute(f"DESCRIBE {name}").fetchall()
        col_names = [c[0] for c in cols]
        print(f"    Columns: {', '.join(col_names[:8])}...")
    conn.close()
