"""
CLI entry point to run incremental BERTopic benchmarks.

Populate `topic_models` with incremental/online BERTopic instances that support
`partial_fit`. The runner streams batches from an IncrementalBERTopicDataset and
records metrics over time via IncrementalBERTopicRunner.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.benchmarks.incremental.reuters_rcv1 import Reuters21578IncrementalDataset
from src.benchmarks.incremental.stackexchange import StackExchangeIncrementalDataset
from src.benchmarks.incremental.tweetner7 import TweetNER7IncrementalDataset
from src.utils.bertopic_utils import (
	IncrementalBERTopicDataset,
	IncrementalBERTopicRunner,
)
from src.utils.bertopic_refit_wrapper import BERTopicRefitWrapper

from src.cobweb.BERTopicIncrementalCobwebWrapper import BERTopicPersistentCobwebWrapper

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from bertopic.dimensionality import BaseDimensionalityReduction
from hdbscan import HDBSCAN
from src.utils.dbstream import DBSTREAMRiver
from sklearn.cluster import MiniBatchKMeans, KMeans
from src.utils.online_lda_wrapper import OnlineLDAWrapper

logger = logging.getLogger(__name__)


class IncrementalBenchmarkRunner:
	"""Dispatch datasets and run incremental BERTopic benchmarks."""

	class RefitCountVectorizer(CountVectorizer):
		"""CountVectorizer that re-fits from scratch on all seen documents each batch."""

		def __init__(self, **kwargs):
			super().__init__(**kwargs)
			self._all_docs: List[str] = []

		def partial_fit(self, raw_documents):
			self._all_docs.extend(list(raw_documents))
			super().fit(self._all_docs)
			return self

		def update_bow(self, raw_documents):
			# Ensure we are fitted on the full corpus seen so far, then transform the new batch.
			if self._all_docs:
				super().fit(self._all_docs)
			return self.transform(raw_documents)

	class FrozenUMAP(UMAP):
		"""UMAP wrapper that fits once and reuses the embedding space on later batches."""

		def __init__(self, **kwargs):
			super().__init__(**kwargs)
			self._fitted = False

		def fit(self, X, y=None):
			# Fit only once; subsequent calls reuse the learned manifold.
			if self._fitted:
				return self
			self._fitted = True
			return super().fit(X, y)

		def partial_fit(self, X, y=None):
			# Fit on the first batch, then freeze the manifold for subsequent calls.
			if not self._fitted:
				self.fit(X, y)
			return self

		def fit_transform(self, X, y=None):
			# First call learns manifold; later calls only transform.
			if not self._fitted:
				self._fitted = True
				return super().fit_transform(X, y)
			return self.transform(X)

	def __init__(self, dataset: str, batch_size: int, first_batch_size: int | None, max_docs: int | None, top_n_words: int, enable_refit: bool, refit_every: int, test_olda: bool):
		self.dataset = dataset.lower()
		self.batch_size = batch_size
		self.first_batch_size = first_batch_size
		self.max_docs = max_docs
		self.top_n_words = top_n_words
		self.enable_refit = enable_refit
		self.refit_every = max(1, refit_every)
		self.test_olda = test_olda

	def _load_dataset(self) -> IncrementalBERTopicDataset:
		if self.dataset in {"reuters", "rcv1", "reuters-21578", "reuters_rcv1"}:
			return Reuters21578IncrementalDataset.load(batch_size=self.batch_size, first_batch_size=self.first_batch_size, max_docs=self.max_docs)
		if self.dataset in {"tweetner7", "tweetner", "tner"}:
			return TweetNER7IncrementalDataset.load(batch_size=self.batch_size, first_batch_size=self.first_batch_size, max_docs=self.max_docs)
		if self.dataset in {"stackexchange", "stack-exchange", "stack"}:
			return StackExchangeIncrementalDataset.load(batch_size=self.batch_size, first_batch_size=self.first_batch_size, max_docs=self.max_docs)
		raise ValueError(f"Unsupported dataset '{self.dataset}'")

	def _build_models(self) -> Tuple[List, List[str]]:
		embedding_model = SentenceTransformer("all-roberta-large-v1")
		# umap_model = self.FrozenUMAP(
		topic_models: List = [
			BERTopic(
				embedding_model=embedding_model,
				umap_model=BaseDimensionalityReduction(),
				hdbscan_model=DBSTREAMRiver(),
				vectorizer_model=self.RefitCountVectorizer(stop_words="english"),
				ctfidf_model=ClassTfidfTransformer(),
			),
			BERTopic(
				embedding_model=embedding_model,
				umap_model=BaseDimensionalityReduction(),
				hdbscan_model=MiniBatchKMeans(n_clusters=7),
				vectorizer_model=self.RefitCountVectorizer(stop_words="english"),
				ctfidf_model=ClassTfidfTransformer(),
			),
			BERTopicRefitWrapper(
				BERTopic(
					embedding_model=embedding_model,
					umap_model=BaseDimensionalityReduction(),
					hdbscan_model=HDBSCAN(min_cluster_size=15, metric="euclidean"),
					vectorizer_model=self.RefitCountVectorizer(stop_words="english"),
					ctfidf_model=ClassTfidfTransformer(),
				),
			),
			BERTopicRefitWrapper(
				BERTopic(
					embedding_model=embedding_model,
					umap_model=BaseDimensionalityReduction(),
					hdbscan_model=KMeans(n_clusters=7),
					vectorizer_model=self.RefitCountVectorizer(stop_words="english"),
					ctfidf_model=ClassTfidfTransformer(),
				),
			),
			BERTopicRefitWrapper(
				BERTopic(
					embedding_model=embedding_model,
					umap_model=BaseDimensionalityReduction(),
					hdbscan_model=BERTopicPersistentCobwebWrapper(max_clusters=12, min_cluster_size=5, leaf_ratio=0.15),
					vectorizer_model=self.RefitCountVectorizer(stop_words="english"),
					ctfidf_model=ClassTfidfTransformer(),
				),
				use_persistent_clusterer=True,
			),
		]

		labels = [
			"DBSTREAM",
			"MiniBatchKMeans",
			"Refit-HDBSCAN",
			"Refit-KMeans",
			"CobwebTM",
		]

		if self.test_olda:
			topic_models.append(OnlineLDAWrapper(num_topics=20, random_state=42))
			labels.append("OnlineLDA")

		return topic_models, labels

	def run(self):
		"""
		Run method on the following benchmarks:
		- Topic coherence (c_v) ↑
		- Topic coherence (NPMI) ↑
		- Topic coherence (c_umass) ↑
		- Topic diversity ↑
		- Intra-topic similarity ↑
		- Inter-topic similarity ↓
		- Topic redundancy ↓
		"""
		dataset = self._load_dataset()
		topic_models, labels = self._build_models()
		if not topic_models:
			raise ValueError("Add incremental BERTopic instances to topic_models before running benchmarks")

		runner = IncrementalBERTopicRunner(topic_models, top_n_words=self.top_n_words)
		results = runner.run(dataset)
		for idx, model_results in enumerate(results):
			logger.info("Model %d produced %d timesteps", idx, len(model_results))

		return results, labels

	@staticmethod
	def plot_metrics(results: List[List[dict]], labels: List[str], output_dir: Path, dataset: str):
		output_dir = output_dir / dataset / "plots"
		output_dir.mkdir(parents=True, exist_ok=True)
		metrics = {
			"coherence_c_v": "Topic coherence (c_v)",
			"coherence_npmi": "Topic coherence (NPMI)",
			"coherence_umass": "Topic coherence (c_umass)",
			"coherence_c_v_batch": "Batch topic coherence (c_v)",
			"coherence_npmi_batch": "Batch topic coherence (NPMI)",
			"coherence_umass_batch": "Batch topic coherence (c_umass)",
			"topic_diversity": "Topic diversity",
			"intra_topic_similarity": "Intra-topic similarity",
			"inter_topic_similarity": "Inter-topic similarity",
			"topic_redundancy": "Topic redundancy",
			"topic_stability_nmi": "Temporal stability (NMI)",
			"topic_stability_ari": "Temporal stability (ARI)",
			"word_overlap_stability": "Word overlap stability",
			"topic_retention_rate": "Topic retention rate",
			"topic_centroid_drift": "Topic centroid drift",
			"topic_word_drift": "Topic word drift",
		}

		for metric_key, title in metrics.items():
			plt.figure(figsize=(8, 5))
			for idx, model_results in enumerate(results):
				y_vals = [step.get(metric_key, float("nan")) for step in model_results]
				x_vals = list(range(1, len(y_vals) + 1))
				plt.plot(x_vals, y_vals, marker="o", label=labels[idx])
			plt.xlabel("Batch index")
			plt.ylabel(title)
			plt.title(title)
			plt.legend()
			plt.grid(True, alpha=0.3)
			plt.tight_layout()
			out_path = output_dir / f"{metric_key}.png"
			plt.savefig(out_path)
			plt.close()

	@staticmethod
	def save_tables(results: List[List[dict]], labels: List[str], output_dir: Path, dataset: str):
		tables_dir = output_dir / dataset / "tables"
		tables_dir.mkdir(parents=True, exist_ok=True)
		metrics = {
			"coherence_c_v": "Topic coherence (c_v)",
			"coherence_npmi": "Topic coherence (NPMI)",
			"coherence_umass": "Topic coherence (c_umass)",
			"coherence_c_v_batch": "Batch topic coherence (c_v)",
			"coherence_npmi_batch": "Batch topic coherence (NPMI)",
			"coherence_umass_batch": "Batch topic coherence (c_umass)",
			"topic_diversity": "Topic diversity",
			"intra_topic_similarity": "Intra-topic similarity",
			"inter_topic_similarity": "Inter-topic similarity",
			"topic_redundancy": "Topic redundancy",
			"topic_stability_nmi": "Temporal stability (NMI)",
			"topic_stability_ari": "Temporal stability (ARI)",
			"word_overlap_stability": "Word overlap stability",
			"topic_retention_rate": "Topic retention rate",
			"topic_centroid_drift": "Topic centroid drift",
			"topic_word_drift": "Topic word drift",
		}

		max_batches = max(len(model_results) for model_results in results)
		for metric_key, title in metrics.items():
			csv_path = tables_dir / f"{metric_key}.csv"
			with csv_path.open("w", newline="") as csvfile:
				writer = csv.writer(csvfile)
				headers = ["batch_index"] + labels
				writer.writerow(headers)
				for batch_idx in range(max_batches):
					row = [batch_idx + 1]
					for model_results in results:
						if batch_idx < len(model_results):
							row.append(model_results[batch_idx].get(metric_key, float("nan")))
						else:
							row.append(float("nan"))
					writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run incremental BERTopic benchmarks")
	parser.add_argument("dataset", help="Dataset to run: reuters_rcv1 | tweetner7 | stackexchange")
	parser.add_argument("--batch-size", type=int, default=512, help="Documents per batch for partial-fit (after the first)")
	parser.add_argument("--first-batch-size", type=int, default=None, help="Optional different size for the first batch")
	parser.add_argument("--max-docs", type=int, default=None, help="Optional limit on documents for quick runs")
	parser.add_argument("--top-n-words", type=int, default=10, help="Top-N words per topic for metrics")
	parser.add_argument("--plot-dir", type=str, default="outputs/incremental_plots", help="Base directory to store per-metric plots (dataset name will be appended)")
	parser.add_argument("--enable-refit", action="store_true", help="Include refit baselines (full-corpus HDBSCAN and persistent Cobweb variants)")
	parser.add_argument("--refit-every", type=int, default=1, help="Refit interval placeholder (currently refits every batch when enabled)")
	parser.add_argument("--test-olda", action="store_true", help="Include an online LDA baseline (gensim) for comparison")
	parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
	return parser.parse_args(argv)


def main(argv: list[str] | None = None):
	args = parse_args(argv)
	logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
	logger.info("Starting incremental benchmark for dataset=%s", args.dataset)
	runner = IncrementalBenchmarkRunner(
		dataset=args.dataset,
		batch_size=args.batch_size,
		first_batch_size=args.first_batch_size,
		max_docs=args.max_docs,
		top_n_words=args.top_n_words,
		enable_refit=args.enable_refit,
		refit_every=args.refit_every,
		test_olda=args.test_olda,
	)
	try:
			results, labels = runner.run()
			plot_dir = Path(args.plot_dir)
			IncrementalBenchmarkRunner.plot_metrics(results, labels, plot_dir, dataset=args.dataset)
			IncrementalBenchmarkRunner.save_tables(results, labels, plot_dir, dataset=args.dataset)
			logger.info("Saved per-batch metric plots to %s", plot_dir / args.dataset / "plots")
			logger.info("Saved per-batch metric tables to %s", plot_dir / args.dataset / "tables")
	except Exception as exc:
		logger.error("Benchmark failed: %s", exc)
		raise


if __name__ == "__main__":
	main(sys.argv[1:])
