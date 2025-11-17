# Narrative Builder (Task 2)

Builds a concise narrative from the provided 84 MB JSON news dataset by:
- filtering to high-rated sources (`source_rating > 8`)
- selecting articles relevant to a user-specified topic
- generating a narrative summary, chronological timeline, thematic clusters, and an optional relationship graph

## Requirements
- Python 3.9+ (stdlib only; no external dependencies)
- Dataset file: `14e9e4cc-9174-48da-ad02-abb1330b48fe.json` (already in this folder)

## Usage
Run from the `narrative_builder_task2` directory.

### Core
```bash
python narrative_builder.py --topic "AI regulation"
python narrative_builder.py --topic "Israel-Iran conflict"
python narrative_builder.py --topic "Jubilee Hills elections"
```

### Options
- `--max-articles N` : cap the number of relevant articles (default: 5).
- `--data-path PATH` : specify a different dataset location (defaults to the provided file).
- `--include-graph`  : include `graph` nodes/edges in the output (omitted by default).

### Example saving output
```bash
python narrative_builder.py --topic "AI regulation" > output_ai.json
```

## Output Format
Consistent JSON structure:
```json
{
  "narrative_summary": "... 5–10 sentences ...",
  "timeline": [
    {
      "date": "...",
      "headline": "...",
      "url": "...",
      "why_it_matters": "..."
    }
  ],
  "clusters": [
    {
      "theme": "...",
      "summary": "...",
      "articles": [
        { "headline": "...", "url": "...", "date": "..." }
      ]
    }
  ],
  "graph": {
    "nodes": [...],
    "edges": [...]
  }
}
```
`graph` is only present when `--include-graph` is used.

## Design Notes
- Token-based relevance ranking favors overlap with the topic phrase and applies stricter gating for short topics.
- If no articles meet the strict overlap threshold, the selector relaxes to top-scoring partial matches to avoid empty narratives.
- Timeline is sorted by published date; clusters group by frequent keywords.
- Graph edges connect articles with shared tokens; relation type is a simple heuristic (`builds_on` or `adds_context`).

## Project Structure
- `narrative_builder.py` — core script (argument parsing, filtering, clustering, timeline, optional graph).
- `14e9e4cc-9174-48da-ad02-abb1330b48fe.json` — provided news dataset (84 MB).
- `README.md` — this guide.
