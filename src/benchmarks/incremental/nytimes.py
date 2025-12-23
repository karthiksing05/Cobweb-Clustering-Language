from __future__ import annotations

from typing import List, Optional

from src.utils.bertopic_utils import IncrementalBERTopicDataset


class NYTimesAnnotatedIncrementalDataset:
    """Load NYTimes Annotated Corpus in temporal order for incremental BERTopic."""

    @classmethod
    def load(
        cls,
        hf_name: str = "nytimes_annotated_corpus",
        hf_config: Optional[str] = None,
        split: str = "train",
        text_fields: Optional[List[str]] = None,
        sort_field_candidates: Optional[List[str]] = None,
        batch_size: int = 512,
        first_batch_size: Optional[int] = None,
        max_docs: Optional[int] = None,
        analyzer: Optional[callable] = None,
    ) -> IncrementalBERTopicDataset:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("NYTimes loader requires the 'datasets' package: pip install datasets") from exc

        candidates = [
            {"path": hf_name, "name": hf_config, "kwargs": {"split": split}},
            {"path": "GEM/nyt", "name": None, "kwargs": {"split": split}},
            {"path": "jfinkels/nytimes", "name": None, "kwargs": {"split": split}},
        ]

        ds = None
        last_err = None
        for cand in candidates:
            if not cand["path"]:
                continue
            try:
                ds = load_dataset(cand["path"], name=cand["name"], **cand["kwargs"])
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc

        if ds is None:
            raise RuntimeError(
                "No accessible NYTimes dataset found; set hf_name/hf_config to a valid source or install the data locally"
            ) from last_err

        sort_fields = sort_field_candidates or [
            "date",
            "publication_date",
            "publish_date",
            "pub_date",
            "created_at",
            "year",
        ]
        for field in sort_fields:
            if field in ds.column_names:
                ds = ds.sort(field)
                break

        default_text_fields = text_fields or ["text", "article", "document", "body", "content"]
        title_fields = ["title", "headline", "section_title", "headline_main"]
        abstract_fields = ["abstract", "summary", "lead", "lede"]

        docs: List[str] = []
        for row in ds:
            doc_title = cls._first_non_empty(row, title_fields)
            abstract = cls._first_non_empty(row, abstract_fields)
            body = cls._first_non_empty(row, default_text_fields)
            parts = [part for part in (doc_title, abstract, body) if part]
            if not parts:
                continue
            docs.append("\n\n".join(parts).strip())

        if max_docs is not None:
            docs = docs[:max_docs]

        if not docs:
            raise ValueError("No documents extracted from the NYTimes dataset candidate")

        return IncrementalBERTopicDataset.from_texts(
            documents=docs,
            batch_size=batch_size,
            first_batch_size=first_batch_size,
            analyzer=analyzer,
        )

    @staticmethod
    def _first_non_empty(row: object, fields: List[str]) -> str:
        if not isinstance(row, dict):
            return ""
        for field in fields:
            val = row.get(field)
            if val:
                return str(val)
        return ""
