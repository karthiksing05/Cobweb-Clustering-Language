"""
Time-sorted TweetNER7 incremental loader for BERTopic experiments.

Uses the tner/tweetner7 dataset and orders tweets by timestamp if available,
otherwise by id/index. Tokens are joined into space-separated text for BERTopic.
"""

from __future__ import annotations

from typing import List, Optional

from src.utils.bertopic_utils import IncrementalBERTopicDataset


class TweetNER7IncrementalDataset:
	"""Load TweetNER7 in temporal order into an IncrementalBERTopicDataset."""

	@classmethod
	def load(
		cls,
		split: str = "train_all",
		batch_size: int = 1,
		first_batch_size: Optional[int] = None,
		max_docs: Optional[int] = None,
		analyzer: Optional[callable] = None,
	) -> IncrementalBERTopicDataset:
		try:
			from datasets import load_dataset
		except ImportError as exc:
			raise ImportError("TweetNER7 loader requires the 'datasets' package: pip install datasets") from exc

		ds = load_dataset("tner/tweetner7", split=split)

		for field in ("timestamp", "created_at", "date", "id"):
			if field in ds.column_names:
				ds = ds.sort(field)
				break

		if "tokens" in ds.column_names:
			docs: List[str] = [" ".join(tokens) for tokens in ds["tokens"]]
		elif "text" in ds.column_names:
			docs = [str(t) for t in ds["text"]]
		else:
			raise ValueError("No tokens/text field found in TweetNER7 dataset")

		if max_docs is not None:
			docs = docs[:max_docs]

		return IncrementalBERTopicDataset.from_texts(
			documents=docs,
			batch_size=batch_size,
			first_batch_size=first_batch_size,
			analyzer=analyzer,
		)
