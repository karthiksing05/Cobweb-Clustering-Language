from __future__ import annotations
from typing import Optional

from src.utils.bertopic_utils import IncrementalBERTopicDataset

class Reuters21578IncrementalDataset:
    """Load Reuters‑21578 in temporal order into an IncrementalBERTopicDataset."""

    @classmethod
    def load(
        cls,
        hf_name: str = "rjjan/reuters21578",
        split: str = "train",
        text_field: str = "text",
        sort_field: str = "date",
           batch_size: int = 512,
           first_batch_size: Optional[int] = None,
        max_docs: Optional[int] = None,
        analyzer: Optional[callable] = None,
    ) -> IncrementalBERTopicDataset:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Loading requires the 'datasets' package: pip install datasets"
            ) from exc

        # Load from a HF dataset that has Parquet/Arrow data
        ds = load_dataset(hf_name, "ModHayes", split=split)

        # Sort if a temporal field is available
        if sort_field in ds.column_names:
            ds = ds.sort(sort_field)

        # Ensure a text field exists
        if text_field not in ds.column_names:
            raise ValueError(f"Column '{text_field}' not found in dataset")

        docs = ds[text_field]
        if max_docs is not None:
            docs = docs[:max_docs]

        return IncrementalBERTopicDataset.from_texts(
            documents=docs,
            batch_size=batch_size,
               first_batch_size=first_batch_size,
            analyzer=analyzer,
        )
