# Semantic Search — Trademarkia AI/ML Task

Lightweight semantic search system over the 20 Newsgroups corpus with fuzzy
clustering, a cluster-partitioned semantic cache, and a FastAPI service.

---

## Project Structure

```
semantic_search/
├── config.py                  # All constants and paths
├── data_loader.py             # Dataset loading + text preprocessing
├── embedder.py                # Sentence embedding (ONNX / sentence-transformers)
├── vector_store.py            # ChromaDB wrapper
├── clustering.py              # UMAP + GMM fuzzy clustering
├── cluster_analysis.py        # Cluster visualisation and reports
├── semantic_cache.py          # Cluster-partitioned semantic cache (pure Python)
├── threshold_explorer.py      # Empirical τ threshold analysis
├── result_engine.py           # Cache-miss result computation
├── models.py                  # Pydantic API schemas
├── dependencies.py            # FastAPI dependency injection
├── app.py                     # FastAPI service
│
├── pipeline_component1.py     # Run: embed corpus + populate ChromaDB
├── pipeline_component2.py     # Run: UMAP + GMM clustering
├── pipeline_component3.py     # Run: cache smoke test + threshold exploration
│
├── requirements.txt
├── .env
├── Dockerfile
└── docker-compose.yml
```

---

## Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Run Order

Each pipeline must be run in order — each depends on the previous.

### Component 1 — Embed corpus + populate ChromaDB
```bash
python pipeline_component1.py
```
Outputs: `data/embeddings.npy`, `chroma_store/`

### Component 2 — Fuzzy clustering
```bash
python pipeline_component2.py
```
Outputs: `data/gmm_model.pkl`, `data/umap_model.pkl`,
`data/soft_assignments.npy`, `logs/cluster_analysis/`

Flags:
```bash
python pipeline_component2.py --refit         # redo everything
python pipeline_component2.py --refit-gmm     # redo GMM only
python pipeline_component2.py --skip-analysis # skip plots
```

### Component 3 — Validate cache + threshold exploration
```bash
python pipeline_component3.py
```
Outputs: `logs/threshold_explorer/`

### Component 4 — Start the API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Semantic query with cache lookup |
| `GET` | `/cache/stats` | Cache hit/miss statistics |
| `DELETE` | `/cache` | Flush cache and reset stats |
| `PATCH` | `/cache/threshold` | Update τ without restart |
| `GET` | `/health` | Liveness check |

Interactive docs: **http://localhost:8000/docs**

### POST /query — example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What caused the Challenger shuttle disaster?"}'
```

**Cache miss response:**
```json
{
  "query": "What caused the Challenger shuttle disaster?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.712,
  "result": "Top 5 results for: 'What caused the Challenger...'",
  "dominant_cluster": 4,
  "retrieved_docs": [...]
}
```

**Cache hit response (same query rephrased):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Why did the Challenger space shuttle explode?"}'
```
```json
{
  "query": "Why did the Challenger space shuttle explode?",
  "cache_hit": true,
  "matched_query": "What caused the Challenger shuttle disaster?",
  "similarity_score": 0.913,
  "result": "Top 5 results for: 'What caused the Challenger...'",
  "dominant_cluster": 4
}
```

### GET /cache/stats
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405,
  "threshold": 0.85,
  "partitions": {"4": 12, "7": 9, "11": 21}
}
```

### DELETE /cache
```json
{
  "message": "Cache flushed successfully.",
  "entries_cleared": 42
}
```

---

## Docker

```bash
# Build and run
docker-compose up --build

# Or plain Docker
docker build -t semantic-search .
docker run -p 8000:8000 semantic-search
```

> **Note:** Run Component 1 and 2 pipelines **before** building the Docker
> image. The container does not regenerate embeddings or models — it only
> serves the pre-built artefacts from `data/` and `chroma_store/`.

---

## Similarity Threshold τ

The cache's core tunable parameter. See `logs/threshold_explorer/` after
running Component 3 for a full empirical analysis.

| τ | Behaviour |
|---|-----------|
| 0.50 | Dangerous — unrelated queries share results |
| 0.75 | Too lenient — adjacent topics collide |
| **0.85** | **Default — paraphrases match, distinct queries don't** |
| 0.92 | Too strict — paraphrases miss |
| 0.98 | Effectively useless |

Update live without restart:
```bash
curl -X PATCH "http://localhost:8000/cache/threshold?threshold=0.88"
```