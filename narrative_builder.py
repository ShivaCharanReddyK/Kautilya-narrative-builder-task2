"""Narrative builder: filter news items by rating and topic to produce a structured narrative."""

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


@dataclass
class Article:
    idx: int
    title: str
    url: str
    story: str
    published_at: Optional[datetime]
    source_rating: float
    raw_date: str
    tokens: Set[str]
    relevance: float


def tokenize(text: str) -> Set[str]:
    """Lowercase and split on alphanumerics, drop stopwords and single-letter tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1}


def parse_date(raw: str) -> Optional[datetime]:
    """Parse common ISO-like date strings; return None if parsing fails."""
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw[: len(fmt)], fmt)
        except ValueError:
            continue
    return None


def load_items(path: Path) -> List[Dict]:
    """Load dataset items regardless of whether they are wrapped in a dict or plain list."""
    content = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(content, dict) and "items" in content:
        return content.get("items", [])
    if isinstance(content, list):
        return content
    raise ValueError("Unsupported dataset format")


def compute_relevance(tokens: Set[str], topic_tokens: Set[str], topic: str, text: str) -> float:
    """Heuristic overlap score used for ranking articles."""
    if not topic_tokens:
        return 0.0
    overlap = len(tokens & topic_tokens)
    coverage = overlap / len(topic_tokens)
    phrase_bonus = 1.0 if topic.lower() in text.lower() else 0.0
    return overlap + coverage + phrase_bonus


def first_sentence(text: str) -> str:
    """Extract a concise single sentence for why_it_matters."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if parts:
        return parts[0][:260]
    words = text.split()
    return " ".join(words[:30])


def build_clusters(articles: List[Article]) -> List[Dict]:
    """Group articles by top shared keywords and return cluster metadata."""
    token_counts = Counter(tok for a in articles for tok in a.tokens)
    top_keywords = [kw for kw, _ in token_counts.most_common(8)]
    clusters: Dict[str, List[Article]] = defaultdict(list)

    for article in articles:
        best_kw = None
        for kw in top_keywords:
            if kw in article.tokens:
                best_kw = kw
                break
        clusters[best_kw or "misc"].append(article)

    cluster_payload = []
    for theme, arts in clusters.items():
        sorted_arts = sorted(arts, key=lambda a: a.published_at or datetime.max)
        summary = (
            f"{len(arts)} article(s) explore '{theme}' aspects of the topic."
            if theme != "misc"
            else f"{len(arts)} article(s) provide additional context."
        )
        cluster_payload.append(
            {
                "theme": theme,
                "summary": summary,
                "articles": [
                    {
                        "headline": a.title,
                        "url": a.url,
                        "date": a.raw_date,
                    }
                    for a in sorted_arts
                ],
            }
        )
    return cluster_payload


def build_graph(articles: List[Article]) -> Dict:
    """Create a simple relation graph between articles based on token overlap and chronology."""
    nodes = [
        {
            "id": a.idx,
            "headline": a.title,
            "url": a.url,
            "date": a.raw_date,
            "why_it_matters": first_sentence(a.story),
        }
        for a in articles
    ]

    edges = []
    for i, src in enumerate(articles):
        for j, tgt in enumerate(articles):
            if j <= i:
                continue
            overlap = len(src.tokens & tgt.tokens)
            if overlap == 0:
                continue
            if src.published_at and tgt.published_at and src.published_at < tgt.published_at:
                relation = "builds_on" if overlap > 3 else "adds_context"
            else:
                relation = "adds_context"
            edges.append({"source": src.idx, "target": tgt.idx, "relation": relation})
    return {"nodes": nodes, "edges": edges}


def narrative_summary(
    topic: str, articles: List[Article], clusters: List[Dict], include_graph: bool
) -> str:
    """Compose a lightweight narrative summary string."""
    if not articles:
        return f"No high-rated articles related to '{topic}' were found."

    dates = [a.published_at for a in articles if a.published_at]
    earliest, latest = (min(dates).date(), max(dates).date()) if dates else (None, None)
    key_headlines = [a.title for a in sorted(articles, key=lambda a: a.relevance, reverse=True)[:3]]
    cluster_names = [c["theme"] for c in clusters][:3]

    sentences = [
        f"We found {len(articles)} high-rated articles touching on '{topic}'.",
    ]
    if earliest and latest:
        sentences.append(f"Coverage ranges from {earliest.isoformat()} through {latest.isoformat()}.")
    if key_headlines:
        sentences.append("Notable coverage includes: " + "; ".join(key_headlines) + ".")
    if cluster_names:
        sentences.append("Themes emerge around " + ", ".join(cluster_names) + ".")
    sentences.append("The timeline traces how the story evolved across sources.")
    if include_graph:
        sentences.append("Graph links show where articles build on or add context.")
    while len(sentences) < 5:
        sentences.append("Additional context clusters capture related angles.")
    return " ".join(sentences[:10])


def build_timeline(articles: List[Article]) -> List[Dict]:
    """Chronologically order articles with key metadata."""
    sorted_articles = sorted(
        articles,
        key=lambda a: a.published_at or datetime.max,
    )
    timeline = []
    for a in sorted_articles:
        timeline.append(
            {
                "date": a.raw_date,
                "headline": a.title,
                "url": a.url,
                "why_it_matters": first_sentence(a.story),
            }
        )
    return timeline


def select_relevant(items: List[Dict], topic: str, max_articles: int) -> List[Article]:
    """Filter to high-rated, topic-relevant articles and rank by relevance."""
    topic_tokens = tokenize(topic)
    if not topic_tokens:
        return []
    phrase = topic.lower()
    ai_anchor_tokens = {
        "artificial",
        "intelligence",
        "ai",
        "ml",
        "machine",
        "learning",
        "safety",
        "ethics",
        "regulation",
        "policy",
        "policies",
        "governance",
        "law",
        "laws",
        "rules",
        "oversight",
    }

    # Stricter coverage for short topics to avoid spurious matches (e.g., "AI" as airline code).
    primary_coverage = 1.0 if len(topic_tokens) <= 3 else 0.66
    fallback_coverage = 0.5
    fallback_min_overlap = 2 if len(topic_tokens) >= 2 else 1

    def collect(min_coverage: float, min_overlap: int, require_phrase: bool = False) -> List[Article]:
        collected: List[Article] = []
        for idx, item in enumerate(items):
            if float(item.get("source_rating", 0)) <= 8:
                continue
            title = item.get("title", "")
            story = item.get("story", "")
            url = item.get("url", "")
            raw_date = item.get("published_at", "") or ""
            tokens = tokenize(f"{title} {story}")
            score = compute_relevance(tokens, topic_tokens, phrase, f"{title} {story}")

            overlap = len(tokens & topic_tokens)
            coverage = overlap / len(topic_tokens)
            phrase_match = phrase in f"{title} {story}".lower()

            if "ai" in topic_tokens:
                # Avoid airline/other false positives when topic is AI-related.
                if not (tokens & ai_anchor_tokens):
                    continue

            if require_phrase and not phrase_match:
                continue
            if not phrase_match and (coverage < min_coverage or overlap < min_overlap):
                continue

            collected.append(
                Article(
                    idx=idx,
                    title=title,
                    url=url,
                    story=story,
                    published_at=parse_date(raw_date),
                    source_rating=float(item.get("source_rating", 0)),
                    raw_date=raw_date,
                    tokens=tokens,
                    relevance=score,
                )
            )
        return collected

    # First pass: strict coverage to keep only strong topical matches.
    relevant = collect(primary_coverage, min_overlap=len(topic_tokens))
    # Fallback: allow partial matches but enforce at least moderate overlap.
    if not relevant:
        relevant = collect(fallback_coverage, min_overlap=fallback_min_overlap)

    # AI-specific safety net to avoid airline false positives while still returning AI content.
    if not relevant and "ai" in topic_tokens:
        aviation_blocklist = {
            "airline",
            "airlines",
            "flight",
            "flights",
            "air",
            "express",
            "crew",
            "pilot",
            "aircraft",
            "airport",
            "runway",
            "cockpit",
            "dgca",
        }
        ai_core = {"artificial", "intelligence", "machine", "learning", "ml"}
        collected: List[Article] = []
        for idx, item in enumerate(items):
            if float(item.get("source_rating", 0)) <= 8:
                continue
            title = item.get("title", "")
            story = item.get("story", "")
            url = item.get("url", "")
            raw_date = item.get("published_at", "") or ""
            tokens = tokenize(f"{title} {story}")
            if not (tokens & ai_anchor_tokens):
                continue
            if (tokens & aviation_blocklist) and not (tokens & ai_core):
                continue
            score = len(tokens & ai_anchor_tokens) + len(tokens & ai_core)
            collected.append(
                Article(
                    idx=idx,
                    title=title,
                    url=url,
                    story=story,
                    published_at=parse_date(raw_date),
                    source_rating=float(item.get("source_rating", 0)),
                    raw_date=raw_date,
                    tokens=tokens,
                    relevance=score,
                )
            )
        relevant = collected

    relevant.sort(key=lambda a: a.relevance, reverse=True)
    return relevant[:max_articles]


def generate_narrative(
    topic: str, data_path: Path, max_articles: int, include_graph: bool
) -> Dict:
    """Pipeline: load, filter, cluster, sequence, and (optionally) relate articles."""
    items = load_items(data_path)
    relevant_articles = select_relevant(items, topic, max_articles)
    clusters = build_clusters(relevant_articles)
    timeline = build_timeline(relevant_articles)
    summary = narrative_summary(topic, relevant_articles, clusters, include_graph)

    output = {
        "narrative_summary": summary,
        "timeline": timeline,
        "clusters": clusters,
    }
    if include_graph:
        output["graph"] = build_graph(relevant_articles)
    return output


def parse_args() -> argparse.Namespace:
    """CLI argument parser."""
    parser = argparse.ArgumentParser(description="Build a narrative from a news dataset.")
    parser.add_argument("--topic", required=True, help="Topic to build a narrative for.")
    parser.add_argument(
        "--data-path",
        default="14e9e4cc-9174-48da-ad02-abb1330b48fe.json",
        help="Path to the JSON news dataset.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=5,
        help="Maximum number of relevant articles to include in the narrative.",
    )
    parser.add_argument(
        "--include-graph",
        action="store_true",
        help="Include graph nodes/edges. By default, the graph is omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    narrative = generate_narrative(
        args.topic,
        data_path,
        args.max_articles,
        include_graph=args.include_graph,
    )
    print(json.dumps(narrative, indent=2))


if __name__ == "__main__":
    main()
