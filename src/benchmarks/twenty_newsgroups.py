"""
20 Newsgroups dataset loader compatible with BERTopic experiments.

Requires scikit-learn's 20 Newsgroups fetcher. Optionally trims headers,
footers, and quotes to focus on message bodies.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from sklearn.datasets import fetch_20newsgroups

from src.utils.bertopic_utils import BERTopicDataset


class TwentyNewsgroupsDataset(BERTopicDataset):
	"""Load 20 Newsgroups into a BERTopicDataset."""

	@classmethod
	def load(
		cls,
		subset: str = "test",
		categories: Optional[Sequence[str]] = None,
		remove_headers: bool = True,
		max_docs: Optional[int] = None,
		analyzer: Optional[callable] = None,
	) -> "TwentyNewsgroupsDataset":
		remove = ("headers", "footers", "quotes") if remove_headers else ()
		dataset = fetch_20newsgroups(subset=subset, categories=categories, remove=remove)
		docs: List[str] = dataset.data
		if max_docs is not None:
			docs = docs[:max_docs]
		return cls.from_texts(docs, analyzer=analyzer)
