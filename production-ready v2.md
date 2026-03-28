# RAG Documentation Agent – Production-Ready v2

## 1. Cel systemu

Agent do analizy dokumentacji technicznej (np. Red Hat, OpenShift, Linux), który:

* odpowiada na pytania na podstawie źródeł
* zwraca cytaty (źródła + sekcje)
* rozróżnia wersje dokumentacji
* działa w środowisku air-gapped
* jest ewaluowalny (quality measurable)

---

## 2. Architektura (wysoki poziom)

Pipeline:

1. Ingest (scraper / loader)
2. Parsing (HTML → struktura)
3. Chunking (semantyczny)
4. Metadata enrichment
5. Embedding
6. Indexing (vector + keyword)
7. Retrieval (hybrid)
8. Reranking
9. Answer generation + citations

---

## 3. Ingest / źródła

### Źródła:

* HTML (priorytet)
* Markdown
* PDF (fallback – parsowany do tekstu)

### Strategia aktualizacji:

* hash dokumentu (SHA256)
* last_seen_at
* diff detection (jeśli hash się zmieni → re-index tylko danego dokumentu)

---

## 4. Chunking

### Zasady:

* chunk per sekcja (H1/H2/H3)
* max 500–800 tokenów
* overlap: 10–20%

### Dodatkowe:

* nie przecinaj code blocków
* zachowuj kontekst (nagłówki nadrzędne)

---

## 5. METADATA (KRYTYCZNE – v2 upgrade)

Każdy chunk MUSI zawierać:

* product (np. openshift, rhel)
* major_version (np. 4, 8)
* minor_version (np. 4.14)
* doc_type (admin_guide, networking, security, release_notes)
* section_title
* section_id
* source_url
* source_hash
* last_seen_at
* language

Opcjonalne (ale zalecane):

* tags (np. "storage", "ovn", "upgrade")
* importance_score (manual/auto)

---

## 6. Storage

### Start (POC / v1):

* LanceDB (embedded)
* lokalny filesystem

### Production:

* Qdrant
* lub PostgreSQL + pgvector (jeśli wymagania organizacyjne)

### Keyword search:

* BM25 (np. tantivy / elastic-lite / sqlite fts)

---

## 7. Embeddings

Wymagania:

* model lokalny lub API
* stabilność między wersjami

Rekomendacja:

* jeden model embeddings (nie miksować na start)

---

## 8. Retrieval pipeline (UPGRADE v2)

1. Query → embedding
2. Vector search (top-k=20)
3. Keyword search (top-k=20)
4. Merge (union + dedup)
5. Metadata filtering (np. wersja)
6. RERANKER (top-k=5–10)

---

## 9. RERANKER (NOWY ELEMENT – KRYTYCZNY)

Cel:

* poprawa trafności wyników

Wejście:

* query + candidate chunks

Wyjście:

* posortowane wyniki wg trafności

Bez tego jakość odpowiedzi spada znacząco.

---

## 10. Answer generation

Agent:

* odpowiada TYLKO na podstawie retrieved context
* musi cytować źródła

### Format odpowiedzi:

* odpowiedź
* lista źródeł (URL + sekcja)

---

## 11. CITATION POLICY (NOWE – KRYTYCZNE)

* każda odpowiedź musi mieć cytaty
* brak źródeł = brak odpowiedzi
* jeśli confidence niski → agent mówi "nie mam pewności"

---

## 12. Eval system (NOWE – KRYTYCZNE)

### Dataset:

* 30–50 pytań
* każde pytanie ma expected source

### Metryki:

* retrieval accuracy
* answer correctness
* citation correctness

Bez eval system → brak kontroli jakości.

---

## 13. Update strategy (UPGRADE v2)

* wykrywanie zmian przez hash
* reindex tylko zmienionych dokumentów
* wersjonowanie danych

---

## 14. Deployment

### Tryby:

* lokalny (air-gap)
* serwerowy

### Stack:

* Python
* CLI + opcjonalny UI

---

## 15. Narzędzia (rekomendacja)

* LLM: Claude (główna logika)
* IDE: Cursor (opcjonalnie)
* DB: LanceDB → Qdrant

---

## 16. Roadmap

### v1:

* ingest
* chunking
* basic search

### v2 (ten dokument):

* metadata
* reranker
* eval
* citations

### v3:

* multi-agent
* diff między wersjami
* proactive suggestions

---

## 17. Największe ryzyka

* brak metadata → słaby retrieval
* brak rerankera → niski precision
* brak eval → brak kontroli jakości

---

## 18. Finalny werdykt

Ten dokument definiuje system gotowy do budowy rozwiązania produkcyjnego.
Najważniejsze elementy v2:

* metadata
* reranker
* eval
* citation policy
* incremental updates
