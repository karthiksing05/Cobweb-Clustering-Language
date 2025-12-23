"""
CLI entry point to run BERTopic benchmarks across common datasets.

Populate `topic_models` with BERTopic instances before running. The runner will
load the requested dataset, fit each model, and report the metrics produced by
`BERTopicRunner`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List

from src.benchmarks.ag_news import AGNewsDataset
from src.benchmarks.reuters_21578 import Reuters21578Dataset
from src.benchmarks.twenty_newsgroups import TwentyNewsgroupsDataset
from src.utils.bertopic_utils import BERTopicRunner
from src.utils.hierarchical_utils import BERTopicHierarchicalRunner

from src.cobweb.BERTopicCobwebWrapper import BERTopicCobwebWrapper

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

logger = logging.getLogger(__name__)


class BenchmarkRunner:
	"""Dispatch datasets and run BERTopic benchmarks."""

	def __init__(self, dataset: str, max_docs: int | None, top_n_words: int, run_hierarchical: bool = False, leaf_level_zero: bool = True, reverse_levels: bool = False):
		self.dataset = dataset.lower()
		self.max_docs = max_docs
		self.top_n_words = top_n_words
		self.run_hierarchical = run_hierarchical
		self.leaf_level_zero = leaf_level_zero
		self.reverse_levels = reverse_levels

	def _load_dataset(self):
		if self.dataset in {"20newsgroups", "newsgroups"}:
			return TwentyNewsgroupsDataset.load(max_docs=self.max_docs)
		if self.dataset in {"reuters", "reuters21578", "reuters-21578"}:
			return Reuters21578Dataset.load(max_docs=self.max_docs)
		if self.dataset in {"ag", "agnews", "ag-news"}:
			return AGNewsDataset.load(max_docs=self.max_docs)
		raise ValueError(f"Unsupported dataset '{self.dataset}'")

	def _build_models(self):
		"""
        PRIMARY METHOD FOR US TO CHANGE MODEL PARAMETERS - we're going to be running the best versions of
		everything so once we freeze parameters there won't be anything to pass in customizably through
		the command line. We'll start by initializing the same UMAP for everything with different KMeans
		and DBSCAN model types, and then extend to compare to DBSTREAM for an incremental benchmark.
        """
		embedding_model = SentenceTransformer("all-roberta-large-v1")
		umap_model = UMAP(n_neighbors=15, n_components=128, metric='cosine')
		vectorizer_model = CountVectorizer(stop_words="english")
		ctfidf_model = ClassTfidfTransformer()

		topic_models = [
			# HDBSCAN!!
			BERTopic(
				embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=HDBSCAN(min_cluster_size=15, metric="euclidean"),    
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model
            ),
			# KMeans!!
			BERTopic(
				embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=KMeans(n_clusters=50),    
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model
            ),
			# Cobweb!!
			BERTopic(
				embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=BERTopicCobwebWrapper(cluster_level=5, min_cluster_size=5),
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model
            )
        ]
		return topic_models

	def run(self):
		dataset = self._load_dataset()
		topic_models = self._build_models()
		if not topic_models:
			raise ValueError("Add BERTopic instances to topic_models before running benchmarks")

		runner = BERTopicRunner(topic_models)
		results = runner.run(dataset, top_n_words=self.top_n_words)
		for idx, metrics in enumerate(results):
			model = metrics.pop("model")
			logger.info("Model %d (%s) metrics: %s", idx, model.hdbscan_model.__class__.__name__, metrics)
			print(f"Model {idx} ({model.hdbscan_model.__class__.__name__}) metrics: {metrics}")
		if self.run_hierarchical:
			# Reuse the trained model instances returned by the non-hierarchical runner
			# trained_models = [entry["model"] for entry in results]
			hierarchical_runner = BERTopicHierarchicalRunner(
				topic_models,
				leaf_level_zero=self.leaf_level_zero,
				reverse_levels=self.reverse_levels,
			)
			hierarchical_results = hierarchical_runner.run(dataset, top_n_words=self.top_n_words)
			for idx, metrics in enumerate(hierarchical_results):
				model = metrics.pop("model")
				logger.info("Hierarchical Model %d (%s) metrics: %s", idx, model.hdbscan_model.__class__.__name__, metrics)
				print(f"Hierarchical Model {idx} ({model.hdbscan_model.__class__.__name__}) metrics: {metrics}")
		return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run BERTopic benchmarks")
	parser.add_argument("dataset", help="Dataset to run: 20newsgroups | reuters | ag_news")
	parser.add_argument("--max-docs", type=int, default=None, help="Optional limit on documents for quick runs")
	parser.add_argument("--top-n-words", type=int, default=15, help="Top-N words per topic for metrics")
	parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
	parser.add_argument("--test_hierarchical", default=True, action="store_true", help="Whether to test hierarchical clustering models")
	parser.add_argument("--leaf_level_zero", default=True, action="store_true", help="Align hierarchy reporting so leaves map to Level 0 (default)")
	parser.add_argument("--reverse_levels", default=False, action="store_true", help="Report hierarchical metrics from parents down to leaves")
	return parser.parse_args(argv)


def main(argv: list[str] | None = None):
	args = parse_args(argv)
	logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
	logger.info("Starting benchmark for dataset=%s", args.dataset)
	runner = BenchmarkRunner(
		dataset=args.dataset,
		max_docs=args.max_docs,
		top_n_words=args.top_n_words,
		run_hierarchical=args.test_hierarchical,
		leaf_level_zero=args.leaf_level_zero,
		reverse_levels=args.reverse_levels,
	)
	try:
		runner.run()
	except Exception as exc:
		logger.error("Benchmark failed: %s", exc)
		raise


if __name__ == "__main__":
	main(sys.argv[1:])
