"""
Time-sorted StackOverflow/StackExchange incremental loader for BERTopic experiments.

Attempts multiple Hugging Face datasets (e.g., `pacovaldez/stackoverflow-questions`,
`c17hawke/stackoverflow-dataset`) and orders entries by creation timestamp when
available, otherwise by id/index.
"""

from __future__ import annotations

from typing import List, Optional

from src.utils.bertopic_utils import IncrementalBERTopicDataset


class StackExchangeIncrementalDataset:
	"""Load StackExchange posts in temporal order into an IncrementalBERTopicDataset."""

	@classmethod
	def load(
		cls,
		split: str = "train",
		batch_size: int = 512,
		first_batch_size: Optional[int] = None,
		max_docs: Optional[int] = None,
		analyzer: Optional[callable] = None,
	) -> IncrementalBERTopicDataset:
		try:
			from datasets import load_dataset
		except ImportError as exc:
			raise ImportError("StackExchange loader requires the 'datasets' package: pip install datasets") from exc

		# Try several known dataset names/configs
		candidates = [
			{"path": "pacovaldez/stackoverflow-questions", "kwargs": {"split": split}},
			{"path": "c17hawke/stackoverflow-dataset", "kwargs": {"split": split}},
		]

		ds = None
		last_err = None
		for cand in candidates:
			try:
				ds = load_dataset(cand["path"], **cand["kwargs"])
				break
			except Exception as exc:  # noqa: BLE001
				last_err = exc

		if ds is None:
			raise RuntimeError("No accessible StackOverflow/StackExchange dataset found on Hugging Face") from last_err

		for field in ("CreationDate", "creation_date", "creation", "timestamp", "date", "Id", "id"):
			if field in ds.column_names:
				ds = ds.sort(field)
				break

		docs: List[str] = []
		for row in ds:
			title = row.get("Title", "") if isinstance(row, dict) else ""
			if not title and isinstance(row, dict):
				title = row.get("title", "")
			body = row.get("Body", "") if isinstance(row, dict) else ""
			if not body and isinstance(row, dict):
				body = row.get("body", "")
			question = (title or "") + "\n" + (body or "")
			docs.append(question.strip())

		if max_docs is not None:
			docs = docs[:max_docs]

		return IncrementalBERTopicDataset.from_texts(
			documents=docs,
			batch_size=batch_size,
			first_batch_size=first_batch_size,
			analyzer=analyzer,
		)
