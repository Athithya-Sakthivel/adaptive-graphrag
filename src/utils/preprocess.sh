

cd /workspace
python3 - <<'PYEOF'
"""Rebuild clusters_filtered.parquet with the original schema's dtypes."""
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

path = Path("tmp/data/raw/metadata/clusters_filtered.parquet")
bak  = path.with_suffix(".parquet.bak")

if not bak.exists():
    raise SystemExit(f"backup missing: {bak} — cannot recover original schema")

# Read the ORIGINAL parquet, preserving its exact schema
original = pq.read_table(bak)
orig_schema = original.schema

print("=== original schema ===")
print(orig_schema)

df = original.to_pandas()
before_rows = len(df)
print(f"\nbefore: {before_rows} rows")

# Dedup by id — keep the row with the most non-null fields
df["_score"] = df.notna().sum(axis=1)
df = df.sort_values(["id", "_score"], ascending=[True, False])
df = df.drop_duplicates(subset="id", keep="first")
df = df.drop(columns=["_score"]).reset_index(drop=True)
print(f"after:  {len(df)} rows (dropped {before_rows - len(df)})")

# Rebuild the Arrow table using the ORIGINAL schema, so int64 stays int64
# and dtypes aren't silently promoted by pandas.
new_table = pa.Table.from_pandas(df, schema=orig_schema, preserve_index=False)

print("\n=== rebuilt schema ===")
print(new_table.schema)

pq.write_table(new_table, path)
print(f"\nwrote: {path}")

# Verify: read back and check dtypes
check = pq.read_table(path)
print("\n=== verification ===")
for name in ("id", "citation_count", "docket_id"):
    if name in check.schema.names:
        print(f"  {name}: {check.schema.field(name).type}")

# Sanity check the specific case
row = check.to_pandas()
m = row[row["id"] == 1111422]
print(f"\nid=1111422 ({len(m)} row):")
print(m[["id", "case_name", "date_filed", "citation_count"]].to_string())
PYEOF
