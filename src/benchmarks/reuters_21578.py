"""
Reuters-21578 dataset loader compatible with BERTopic experiments.

Uses the NLTK Reuters corpus. Ensure the corpus is downloaded via
``nltk.download('reuters')`` and the NLTK data path is configured.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from src.utils.bertopic_utils import BERTopicDataset


class Reuters21578Dataset(BERTopicDataset):
	"""Load Reuters-21578 into a BERTopicDataset."""

	@classmethod
	def load(
		cls,
		categories: Optional[Sequence[str]] = None,
		max_docs: Optional[int] = None,
		analyzer: Optional[callable] = None,
	) -> "Reuters21578Dataset":
		try:
			from nltk.corpus import reuters
		except ImportError as exc:
			raise ImportError("Reuters-21578 requires nltk; install nltk and download the 'reuters' corpus") from exc

		try:
			file_ids = reuters.fileids(categories) if categories else reuters.fileids()
		except LookupError as exc:
			raise LookupError("NLTK 'reuters' corpus not found. Run nltk.download('reuters')") from exc

		docs: List[str] = [reuters.raw(fid) for fid in file_ids]
		if max_docs is not None:
			docs = docs[:max_docs]
		return cls.from_texts(docs, analyzer=analyzer)
