"""
Load rows from a CSV file into a ChromaDB collection.
 
Each row becomes one document:
  - document text  = one or more columns joined together (what gets embedded)
  - id             = an ID column, or a stable hash of the row if none is given
  - metadata       = the remaining columns (or a chosen subset), type-coerced
 
Usage:
  pip install chromadb
  python csv_to_chroma.py data.csv --collection products --text-cols name,description --id-col sku
"""
 
import argparse
import csv
import hashlib
from typing import Any
 
import chromadb
 
 
def coerce(value: str) -> Any:
    """Chroma metadata only accepts str/int/float/bool; convert where sensible."""
    v = value.strip()
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v
 
 
def row_id(row: dict) -> str:
    """Deterministic ID so re-running the load upserts instead of duplicating."""
    raw = "|".join(f"{k}={row[k]}" for k in sorted(row))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()
 
 
def load_csv_to_chroma(
    csv_path: str,
    collection_name: str,
    text_cols: list[str],
    id_col: str | None = None,
    metadata_cols: list[str] | None = None,
    persist_dir: str = "./chroma_db",
    batch_size: int = 500,
) -> int:
    client = chromadb.PersistentClient(path=persist_dir)
    # Uses Chroma's default embedding function (all-MiniLM-L6-v2, runs locally).
    collection = client.get_or_create_collection(name=collection_name)
 
    ids, docs, metas = [], [], []
    total = 0
 
    def flush():
        nonlocal total
        if ids:
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            total += len(ids)
            print(f"  upserted {total} rows")
            ids.clear(); docs.clear(); metas.clear()
 
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
 
        missing = [c for c in text_cols + ([id_col] if id_col else []) if c not in headers]
        if missing:
            raise ValueError(f"Columns not found in CSV: {missing}. Available: {headers}")
 
        meta_cols = metadata_cols or [c for c in headers if c not in text_cols and c != id_col]
 
        for row in reader:
            text = "\n".join(f"{c}: {row[c]}" for c in text_cols if row.get(c))
            if not text.strip():
                continue  # nothing to embed
 
            ids.append(str(row[id_col]) if id_col else row_id(row))
            docs.append(text)
            # Skip empty values — Chroma rejects None in metadata.
            meta = {c: coerce(row[c]) for c in meta_cols if row.get(c, "").strip()}
            metas.append(meta or {"source": csv_path})
 
            if len(ids) >= batch_size:
                flush()
        flush()
 
    print(f"Done. Collection '{collection_name}' now has {collection.count()} documents.")
    return total
 
 
def main():
    p = argparse.ArgumentParser(description="Load a CSV into ChromaDB")
    p.add_argument("csv_path")
    p.add_argument("--collection", required=True, help="Chroma collection name")
    p.add_argument("--text-cols", required=True, help="Comma-separated columns to embed")
    p.add_argument("--id-col", help="Column to use as the document ID (default: row hash)")
    p.add_argument("--metadata-cols", help="Comma-separated metadata columns (default: all others)")
    p.add_argument("--persist-dir", default="./chroma_db")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--query", help="Optional: run a test similarity query after loading")
    a = p.parse_args()
 
    load_csv_to_chroma(
        csv_path=a.csv_path,
        collection_name=a.collection,
        text_cols=[c.strip() for c in a.text_cols.split(",")],
        id_col=a.id_col,
        metadata_cols=[c.strip() for c in a.metadata_cols.split(",")] if a.metadata_cols else None,
        persist_dir=a.persist_dir,
        batch_size=a.batch_size,
    )
 
    if a.query:
        col = chromadb.PersistentClient(path=a.persist_dir).get_collection(a.collection)
        res = col.query(query_texts=[a.query], n_results=3)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            print(f"\n[{dist:.3f}] {meta}\n{doc}")
 
 
if __name__ == "__main__":
    main()
 