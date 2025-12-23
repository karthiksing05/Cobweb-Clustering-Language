"""
ML ArXiv Papers dataset loader compatible with BERTopic experiments.

Dataset source: https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers
Downloads via the Hugging Face `datasets` package on first use.
"""

from __future__ import annotations

from typing import List, Optional

from src.utils.bertopic_utils import BERTopicDataset


class MLArXivDataset(BERTopicDataset):
	"""Load ML ArXiv Papers into a BERTopicDataset."""

	@classmethod
	def load(
		cls,
		split: str = "train",
		max_docs: Optional[int] = None,
		analyzer: Optional[callable] = None,
	) -> "MLArXivDataset":
		try:
			from datasets import load_dataset
		except ImportError as exc:
			raise ImportError("ML ArXiv requires the 'datasets' package: pip install datasets") from exc

		dataset = load_dataset("CShorten/ML-ArXiv-Papers", split=split)
		docs: List[str] = [record.get("title", "") + "\n" + record.get("abstract", "") for record in dataset]
		if max_docs is not None:
			docs = docs[:max_docs]
		return cls.from_texts(docs, analyzer=analyzer)
