"""
AG News dataset loader compatible with BERTopic experiments.

Uses the Hugging Face datasets loader. The dataset will be downloaded on first
use if not cached locally.
"""

from __future__ import annotations

from typing import List, Optional

from src.utils.bertopic_utils import BERTopicDataset


class AGNewsDataset(BERTopicDataset):
	"""Load AG News into a BERTopicDataset."""

	@classmethod
	def load(
		cls,
		split: str = "test",
		max_docs: Optional[int] = None,
		analyzer: Optional[callable] = None,
	) -> "AGNewsDataset":
		try:
			from datasets import load_dataset
		except ImportError as exc:
			raise ImportError("AG News requires the 'datasets' package: pip install datasets") from exc

		dataset = load_dataset("ag_news", split=split)
		docs: List[str] = [record["text"] for record in dataset]
		if max_docs is not None:
			docs = docs[:max_docs]
		return cls.from_texts(docs, analyzer=analyzer)
