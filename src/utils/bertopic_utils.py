"""
Utility classes for running and evaluating BERTopic models.

The helpers below wrap dataset loading and multi-model evaluation so experiments
can stay concise in notebooks and scripts. Metrics mirror those commonly used in
the BERTopic ecosystem: c_v coherence, NPMI coherence, topic diversity, and
inter-topic similarity (cosine similarity between topic embeddings).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import jensenshannon

Batch = List[str]

logger = logging.getLogger(__name__)


def _default_analyzer() -> callable:
	"""Return a default analyzer that mirrors CountVectorizer tokenization."""
	vectorizer = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b\w\w+\b")
	return vectorizer.build_analyzer()

@dataclass
class BERTopicDataset:
	"""
	Lightweight dataset wrapper for BERTopic.

	It keeps pre-tokenized documents, a gensim dictionary, and a BoW corpus so
	downstream coherence computations are trivial. Documents are accepted as an
	iterable of strings; embeddings can optionally be precomputed and supplied.
	"""

	documents: List[str]
	tokens: List[List[str]]
	dictionary: Dictionary
	corpus: List[List[Tuple[int, int]]]
	embeddings: Optional[np.ndarray] = None

	@classmethod
	def from_texts(
		cls,
		documents: Iterable[str],
		embeddings: Optional[Sequence[Sequence[float]]] = None,
		analyzer: Optional[callable] = None,
	) -> "BERTopicDataset":
		docs = [doc.strip() for doc in documents if str(doc).strip()]
		if not docs:
			raise ValueError("No documents provided to BERTopicDataset")

		analyzer_fn = analyzer or _default_analyzer()
		tokens = [analyzer_fn(doc) for doc in docs]
		dictionary = Dictionary(tokens)
		corpus = [dictionary.doc2bow(doc_tokens) for doc_tokens in tokens]

		embed_array = None
		if embeddings is not None:
			embed_array = np.asarray(embeddings)
			if embed_array.shape[0] != len(docs):
				raise ValueError("Embeddings and documents must have the same length")

		return cls(documents=docs, tokens=tokens, dictionary=dictionary, corpus=corpus, embeddings=embed_array)

	@classmethod
	def from_text_file(
		cls,
		path: str | Path,
		encoding: str = "utf-8",
		analyzer: Optional[callable] = None,
	) -> "BERTopicDataset":
		text_path = Path(path)
		with text_path.open("r", encoding=encoding) as handle:
			docs = [line.strip() for line in handle if line.strip()]
		return cls.from_texts(docs, analyzer=analyzer)

	@classmethod
	def from_csv(
		cls,
		path: str | Path,
		text_column: str = "text",
		encoding: str = "utf-8",
		analyzer: Optional[callable] = None,
	) -> "BERTopicDataset":
		df = pd.read_csv(path, encoding=encoding)
		if text_column not in df.columns:
			raise ValueError(f"Column '{text_column}' not found in CSV {path}")
		docs = df[text_column].astype(str).tolist()
		embeddings = df["embedding"].tolist() if "embedding" in df.columns else None
		return cls.from_texts(docs, embeddings=embeddings, analyzer=analyzer)

	@property
	def size(self) -> int:
		return len(self.documents)


class BERTopicRunner:
	"""
	Run one or more BERTopic instances on a dataset and compute standard metrics.
	"""

	def __init__(self, topic_models: Sequence[BERTopic]):
		if not topic_models:
			raise ValueError("Provide at least one BERTopic instance to BERTopicRunner")
		self.topic_models = list(topic_models)

	def run(
		self,
		dataset: BERTopicDataset,
		top_n_words: int = 10,
	) -> List[Dict[str, float]]:
		"""
		Fit each model on the dataset and return metric dictionaries.

		Returns a list matching the input models order. Each entry contains the
		fitted model and the requested metrics.
		"""

		results: List[Dict[str, float]] = []
		for model in self.topic_models:
			logger.info("Fitting BERTopic model %s on %d docs", model.hdbscan_model.__class__.__name__, dataset.size)
			model.fit(dataset.documents, embeddings=dataset.embeddings)
			metrics = self._compute_metrics(model, dataset, top_n_words=top_n_words)
			results.append({"model": model, **metrics})
		return results

	def _compute_metrics(
		self,
		model: BERTopic,
		dataset: BERTopicDataset,
		top_n_words: int,
	) -> Dict[str, float]:
		topic_words = self._extract_topic_words(model, top_n=top_n_words)
		coherence_cv = self._coherence(topic_words, dataset, measure="c_v")
		coherence_npmi = self._coherence(topic_words, dataset, measure="c_npmi")
		diversity = self._topic_diversity(topic_words)
		inter_sim = self._inter_topic_similarity(model)

		return {
			"coherence_c_v": coherence_cv,
			"coherence_npmi": coherence_npmi,
			"topic_diversity": diversity,
			"inter_topic_similarity": inter_sim,
		}

	@staticmethod
	def _extract_topic_words(model: BERTopic, top_n: int) -> List[List[str]]:
		topics = model.get_topics()
		if not topics:
			return []

		topic_words: List[List[str]] = []
		for topic_id, words in topics.items():
			if topic_id == -1:
				continue  # skip outlier topic
			topic_words.append([str(word) for word, _ in words[:top_n] if word is not None])
		return topic_words

	@staticmethod
	def _coherence(
		topic_words: List[List[str]],
		dataset: BERTopicDataset,
		measure: str,
	) -> float:
		if not topic_words:
			return float("nan")

		try:
			coherence_model = CoherenceModel(
				topics=topic_words,
				texts=dataset.tokens,
				corpus=dataset.corpus,
				dictionary=dataset.dictionary,
				coherence=measure,
			)
			return float(coherence_model.get_coherence())
		except ValueError:
			logger.warning("Unable to compute %s coherence; returning NaN", measure)
			return float("nan")

	@staticmethod
	def _topic_diversity(topic_words: List[List[str]]) -> float:
		if not topic_words:
			return float("nan")

		unique_terms = set()
		total = 0
		for words in topic_words:
			unique_terms.update(words)
			total += len(words)
		return float(len(unique_terms) / total) if total else float("nan")

	@staticmethod
	def _inter_topic_similarity(model: BERTopic) -> float:
		topics = model.get_topics()
		if not topics or len(topics) < 2:
			return float("nan")

		topic_order = list(topics.keys())
		indices = [i for i, tid in enumerate(topic_order) if tid != -1]

		def _to_dense(matrix):
			return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

		embeddings = getattr(model, "topic_embeddings_", None)
		if embeddings is not None:
			matrix = np.asarray(embeddings)
		elif getattr(model, "c_tf_idf_", None) is not None:
			matrix = _to_dense(model.c_tf_idf_)
		else:
			return float("nan")

		if matrix.shape[0] < 2 or not indices:
			return float("nan")

		filtered = matrix[indices]
		sim = cosine_similarity(filtered)
		upper = sim[np.triu_indices_from(sim, k=1)]
		return float(np.mean(upper)) if upper.size else float("nan")


@dataclass
class IncrementalBERTopicDataset:
	"""
	Dataset wrapper for incremental BERTopic experiments.

	Documents are kept in-memory but exposed as batches to support partial-fit
	workflows. Tokenization and dictionary/corpus generation mirror BERTopicDataset
	so coherence metrics can be computed on seen data at each timestep.
	"""

	documents: List[str]
	tokens: List[List[str]]
	dictionary: Dictionary
	corpus: List[List[Tuple[int, int]]]
	batch_size: int = 512
	first_batch_size: Optional[int] = None
	embeddings: Optional[np.ndarray] = None

	@classmethod
	def from_texts(
		cls,
		documents: Iterable[str],
		batch_size: int = 512,
		first_batch_size: Optional[int] = None,
		embeddings: Optional[Sequence[Sequence[float]]] = None,
		analyzer: Optional[callable] = None,
	) -> "IncrementalBERTopicDataset":
		docs = [doc.strip() for doc in documents if str(doc).strip()]
		if not docs:
			raise ValueError("No documents provided to IncrementalBERTopicDataset")

		analyzer_fn = analyzer or _default_analyzer()
		tokens = [analyzer_fn(doc) for doc in docs]
		dictionary = Dictionary(tokens)
		corpus = [dictionary.doc2bow(doc_tokens) for doc_tokens in tokens]

		embed_array = None
		if embeddings is not None:
			embed_array = np.asarray(embeddings)
			if embed_array.shape[0] != len(docs):
				raise ValueError("Embeddings and documents must have the same length")

		return cls(
			documents=docs,
			tokens=tokens,
			dictionary=dictionary,
			corpus=corpus,
			batch_size=batch_size,
			first_batch_size=first_batch_size,
			embeddings=embed_array,
		)

	def iter_batches(self) -> Iterable[Batch]:
		fb = self.first_batch_size if self.first_batch_size is not None else self.batch_size
		start = 0
		yield self.documents[start : start + fb]
		start += fb
		while start < len(self.documents):
			yield self.documents[start : start + self.batch_size]
			start += self.batch_size

	def iter_token_batches(self) -> Iterable[List[List[str]]]:
		fb = self.first_batch_size if self.first_batch_size is not None else self.batch_size
		start = 0
		yield self.tokens[start : start + fb]
		start += fb
		while start < len(self.tokens):
			yield self.tokens[start : start + self.batch_size]
			start += self.batch_size

	def iter_corpus_batches(self) -> Iterable[List[List[Tuple[int, int]]]]:
		fb = self.first_batch_size if self.first_batch_size is not None else self.batch_size
		start = 0
		yield self.corpus[start : start + fb]
		start += fb
		while start < len(self.corpus):
			yield self.corpus[start : start + self.batch_size]
			start += self.batch_size

	@property
	def size(self) -> int:
		return len(self.documents)

	@property
	def num_batches(self) -> int:
		if self.size == 0:
			return 0
		fb = self.first_batch_size if self.first_batch_size is not None else self.batch_size
		remaining = max(0, self.size - fb)
		return 1 + int(np.ceil(remaining / self.batch_size)) if remaining > 0 else 1


class IncrementalBERTopicRunner:
	"""
	Evaluate incremental BERTopic models over document batches using partial_fit.

	Each timestep fits the next batch, computes topic quality metrics, and
	temporal stability metrics against the previous timestep.
	"""

	def __init__(self, topic_models: Sequence[object], top_n_words: int = 10):
		if not topic_models:
			raise ValueError("Provide at least one incremental BERTopic instance")
		self.topic_models = list(topic_models)
		self.top_n_words = top_n_words

	def run(self, dataset: IncrementalBERTopicDataset) -> List[List[Dict[str, float]]]:
		all_results: List[List[Dict[str, float]]] = [[] for _ in self.topic_models]
		seen_docs: List[str] = []
		seen_tokens: List[List[str]] = []
		seen_corpus: List[List[Tuple[int, int]]] = []
		prev_labels: List[Optional[np.ndarray]] = [None] * len(self.topic_models)
		prev_topics: List[Optional[Dict[int, List[Tuple[str, float]]]]] = [None] * len(self.topic_models)
		prev_c_tf_idf: List[Optional[np.ndarray]] = [None] * len(self.topic_models)
		prev_embeddings: List[Optional[np.ndarray]] = [None] * len(self.topic_models)

		for batch_docs, batch_tokens, batch_corpus in zip(
			dataset.iter_batches(), dataset.iter_token_batches(), dataset.iter_corpus_batches()
		):
			seen_start = len(seen_docs)
			seen_docs.extend(batch_docs)
			seen_tokens.extend(batch_tokens)
			seen_corpus.extend(batch_corpus)

			for idx, model in enumerate(self.topic_models):
				logger.info("Partial fitting model %s on batch starting %d (%d docs)", model.__class__.__name__, seen_start, len(batch_docs))
				model.partial_fit(batch_docs)

				metrics = self._compute_step_metrics(
					model,
					seen_docs,
					seen_tokens,
					seen_corpus,
					prev_topics[idx],
					prev_c_tf_idf[idx],
					prev_embeddings[idx],
					prev_labels[idx],
				)

				prev_topics[idx] = model.get_topics()
				prev_c_tf_idf[idx] = getattr(model, "c_tf_idf_", None)
				prev_embeddings[idx] = getattr(model, "topic_embeddings_", None)
				prev_labels[idx] = metrics["labels_curr"]
				metrics.pop("labels_curr", None)

				all_results[idx].append(metrics)

		return all_results

	def _compute_step_metrics(
		self,
		model: object,
		docs: List[str],
		tokens: List[List[str]],
		corpus: List[List[Tuple[int, int]]],
		prev_topics: Optional[Dict[int, List[Tuple[str, float]]]],
		prev_c_tf_idf: Optional[np.ndarray],
		prev_embeddings: Optional[np.ndarray],
		prev_labels: Optional[np.ndarray],
	) -> Dict[str, float]:
		current_topics = model.get_topics()
		topic_words = self._extract_topic_words(current_topics, top_n=self.top_n_words)
		coherence_cv = self._coherence(topic_words, tokens, corpus, measure="c_v")
		coherence_npmi = self._coherence(topic_words, tokens, corpus, measure="c_npmi")
		coherence_umass = self._coherence(topic_words, tokens, corpus, measure="u_mass")
		diversity = self._topic_diversity(topic_words)
		inter_sim = self._inter_topic_similarity(current_topics, model)
		intra_sim = self._intra_topic_similarity(current_topics, model)
		redundancy = self._topic_redundancy(topic_words)

		# Predict current labels for temporal metrics
		labels_curr = None
		try:
			labels_curr, _ = model.transform(docs)
		except Exception:
			labels_curr = None

		stability = self._temporal_metrics(
			current_topics,
			prev_topics,
			getattr(model, "c_tf_idf_", None),
			prev_c_tf_idf,
			getattr(model, "topic_embeddings_", None),
			prev_embeddings,
			labels_curr,
			prev_labels,
		)

		return {
			"coherence_c_v": coherence_cv,
			"coherence_npmi": coherence_npmi,
			"coherence_umass": coherence_umass,
			"topic_diversity": diversity,
			"intra_topic_similarity": intra_sim,
			"inter_topic_similarity": inter_sim,
			"topic_redundancy": redundancy,
			"labels_curr": labels_curr,
			**stability,
		}

	@staticmethod
	def _extract_topic_words(topics: Dict[int, List[Tuple[str, float]]], top_n: int) -> List[List[str]]:
		if not topics:
			return []
		topic_words: List[List[str]] = []
		for topic_id, words in topics.items():
			if topic_id == -1:
				continue
			topic_words.append([str(word) for word, _ in words[:top_n] if word is not None])
		return topic_words

	@staticmethod
	def _coherence(
		topic_words: List[List[str]],
		tokens: List[List[str]],
		corpus: List[List[Tuple[int, int]]],
		measure: str,
	) -> float:
		# Require at least two non-empty topics and some corpus before coherence.
		filtered_topics = [tw for tw in topic_words if tw]
		if len(filtered_topics) < 2 or not tokens or not corpus:
			logger.debug(
				"Skipping coherence %s: nonempty_topics=%d tokens=%d corpus=%d (raw_topics=%d)",
				measure,
				len(filtered_topics),
				len(tokens),
				len(corpus),
				len(topic_words),
			)
			return float("nan")
		try:
			dictionary = Dictionary(tokens)
			dict_size = len(dictionary)
			if dict_size == 0:
				logger.debug("Skipping coherence %s: empty dictionary", measure)
				return float("nan")
			# Gensim coherence expects a list of tokens per topic. Coerce defensively
			# to plain Python lists of non-empty strings and drop empty topics.
			topics_for_coherence: List[List[str]] = []
			topic_lengths_before = []
			for topic in filtered_topics:
				if topic is None:
					continue
				if not isinstance(topic, (list, tuple)):
					topic_iter = [topic]
				else:
					topic_iter = topic
				topic_lengths_before.append(len(topic_iter))
				clean = [str(tok) for tok in topic_iter if tok is not None and str(tok).strip()]
				if clean:
					topics_for_coherence.append(clean)
			if len(topics_for_coherence) < 2:
				logger.debug(
					"Skipping coherence %s after cleaning topics: nonempty_topics=%d (raw_lengths=%s)",
					measure,
					len(topics_for_coherence),
					topic_lengths_before,
				)
				return float("nan")
			coherence_model = CoherenceModel(
				topics=topics_for_coherence,
				texts=tokens,
				corpus=corpus,
				dictionary=dictionary,
				coherence=measure,
			)
			return float(coherence_model.get_coherence())
		except ValueError as exc:
			logger.warning(
				"Unable to compute %s coherence; returning NaN (%s) [nonempty_topics=%d tokens=%d corpus=%d dict=%d raw_lengths=%s cleaned_lengths=%s sample_topic=%s]",
				measure,
				exc,
				len(filtered_topics),
				len(tokens),
				len(corpus),
				dict_size if 'dict_size' in locals() else -1,
				topic_lengths_before,
				[list(map(len, topics_for_coherence)) if topics_for_coherence else []],
				(topics_for_coherence[0][:5] if topics_for_coherence else []),
			)
			return float("nan")

	@staticmethod
	def _topic_diversity(topic_words: List[List[str]]) -> float:
		if not topic_words:
			return float("nan")
		unique_terms = set()
		total = 0
		for words in topic_words:
			unique_terms.update(words)
			total += len(words)
		return float(len(unique_terms) / total) if total else float("nan")

	@staticmethod
	def _topic_redundancy(topic_words: List[List[str]]) -> float:
		if not topic_words or len(topic_words) < 2:
			return float("nan")
		pairs = 0
		jaccards: List[float] = []
		for i in range(len(topic_words)):
			for j in range(i + 1, len(topic_words)):
				wi, wj = set(topic_words[i]), set(topic_words[j])
				union = len(wi | wj)
				if union == 0:
					continue
				jaccards.append(len(wi & wj) / union)
				pairs += 1
		return float(np.mean(jaccards)) if jaccards else float("nan")

	@staticmethod
	def _intra_topic_similarity(topics: Dict[int, List[Tuple[str, float]]], model: object) -> float:
		# Approximate intra-topic tightness as the fraction of c-TF-IDF mass captured by top words.
		ctfidf = getattr(model, "c_tf_idf_", None)
		if ctfidf is None or not topics:
			return float("nan")
		matrix = ctfidf.toarray() if hasattr(ctfidf, "toarray") else np.asarray(ctfidf)
		topic_ids = [tid for tid in topics.keys() if tid != -1]
		if not topic_ids:
			return float("nan")
		fractions: List[float] = []
		for row_idx, tid in enumerate(topic_ids):
			words = topics[tid]
			top_indices = [idx for idx, _ in enumerate(words)]
			row = matrix[row_idx]
			top_vals = np.sort(row)[-len(top_indices):] if top_indices else []
			total = np.sum(row)
			if total == 0 or len(top_vals) == 0:
				continue
			fractions.append(float(np.sum(top_vals) / total))
		return float(np.mean(fractions)) if fractions else float("nan")

	@staticmethod
	def _inter_topic_similarity(topics: Dict[int, List[Tuple[str, float]]], model: object) -> float:
		if not topics or len(topics) < 2:
			return float("nan")
		matrix = None
		embeddings = getattr(model, "topic_embeddings_", None)
		if embeddings is not None:
			matrix = np.asarray(embeddings)
		elif getattr(model, "c_tf_idf_", None) is not None:
			ctfidf = model.c_tf_idf_
			matrix = ctfidf.toarray() if hasattr(ctfidf, "toarray") else np.asarray(ctfidf)
		else:
			return float("nan")

		topic_ids = [tid for tid in topics.keys() if tid != -1]
		if len(topic_ids) < 2:
			return float("nan")
		filtered = matrix[: len(topic_ids)]
		sim = cosine_similarity(filtered)
		upper = sim[np.triu_indices_from(sim, k=1)]
		return float(np.mean(upper)) if upper.size else float("nan")

	def _temporal_metrics(
		self,
		current_topics: Optional[Dict[int, List[Tuple[str, float]]]],
		prev_topics: Optional[Dict[int, List[Tuple[str, float]]]],
		current_ctfidf: Optional[np.ndarray],
		prev_ctfidf: Optional[np.ndarray],
		current_embeddings: Optional[np.ndarray],
		prev_embeddings: Optional[np.ndarray],
		labels_curr: Optional[np.ndarray],
		labels_prev: Optional[np.ndarray],
	) -> Dict[str, float]:
		if not current_topics:
			return {
				"topic_stability_nmi": float("nan"),
				"topic_stability_ari": float("nan"),
				"word_overlap_stability": float("nan"),
				"topic_retention_rate": float("nan"),
				"topic_centroid_drift": float("nan"),
				"topic_word_drift": float("nan"),
			}

		nmi = float("nan")
		ari = float("nan")
		if labels_prev is not None and labels_curr is not None:
			# Align label vectors to the same length; current batches may include new docs
			# that did not exist at the previous step.
			min_len = min(len(labels_prev), len(labels_curr))
			if min_len > 0:
				nmi = normalized_mutual_info_score(labels_prev[:min_len], labels_curr[:min_len])
				ari = adjusted_rand_score(labels_prev[:min_len], labels_curr[:min_len])

		match = self._match_topics(current_topics, prev_topics, current_embeddings, prev_embeddings, current_ctfidf, prev_ctfidf)
		if not match:
			return {
				"topic_stability_nmi": nmi,
				"topic_stability_ari": ari,
				"word_overlap_stability": float("nan"),
				"topic_retention_rate": float(0.0) if prev_topics else float("nan"),
				"topic_centroid_drift": float("nan"),
				"topic_word_drift": float("nan"),
			}

		word_overlaps: List[float] = []
		centroid_drifts: List[float] = []
		word_drifts: List[float] = []
		for curr_id, prev_id, sim in match:
			curr_words = set([w for w, _ in current_topics[curr_id][: self.top_n_words]])
			prev_words = set([w for w, _ in prev_topics[prev_id][: self.top_n_words]]) if prev_topics else set()
			union = len(curr_words | prev_words)
			if union:
				word_overlaps.append(len(curr_words & prev_words) / union)

			if current_embeddings is not None and prev_embeddings is not None:
				centroid_drifts.append(float(1.0 - sim))

			if current_ctfidf is not None and prev_ctfidf is not None:
				curr_vec = current_ctfidf[curr_id]
				prev_vec = prev_ctfidf[prev_id]
				curr_vec = curr_vec.toarray().ravel() if hasattr(curr_vec, "toarray") else np.asarray(curr_vec).ravel()
				prev_vec = prev_vec.toarray().ravel() if hasattr(prev_vec, "toarray") else np.asarray(prev_vec).ravel()
				# Avoid zero divisions
				if np.sum(curr_vec) > 0 and np.sum(prev_vec) > 0:
					curr_vec = curr_vec / np.sum(curr_vec)
					prev_vec = prev_vec / np.sum(prev_vec)
					word_drifts.append(float(jensenshannon(curr_vec, prev_vec)))

		retention = float(len(match) / len(prev_topics)) if prev_topics else float("nan")

		return {
			"topic_stability_nmi": float(nmi),
			"topic_stability_ari": float(ari),
			"word_overlap_stability": float(np.mean(word_overlaps)) if word_overlaps else float("nan"),
			"topic_retention_rate": retention,
			"topic_centroid_drift": float(np.mean(centroid_drifts)) if centroid_drifts else float("nan"),
			"topic_word_drift": float(np.mean(word_drifts)) if word_drifts else float("nan"),
		}

	@staticmethod
	def _match_topics(
		current_topics: Optional[Dict[int, List[Tuple[str, float]]]],
		prev_topics: Optional[Dict[int, List[Tuple[str, float]]]],
		current_embeddings: Optional[np.ndarray],
		prev_embeddings: Optional[np.ndarray],
		current_ctfidf: Optional[np.ndarray],
		prev_ctfidf: Optional[np.ndarray],
	) -> List[Tuple[int, int, float]]:
		if not current_topics or not prev_topics:
			return []
		curr_ids = [tid for tid in current_topics.keys() if tid != -1]
		prev_ids = [tid for tid in prev_topics.keys() if tid != -1]
		if not curr_ids or not prev_ids:
			return []

		def _matrix(source, target):
			if source is None or target is None:
				return None
			src = source.toarray() if hasattr(source, "toarray") else np.asarray(source)
			tgt = target.toarray() if hasattr(target, "toarray") else np.asarray(target)
			return cosine_similarity(src, tgt)

		sim_mat = None
		if current_embeddings is not None and prev_embeddings is not None:
			sim_mat = cosine_similarity(current_embeddings, prev_embeddings)
		elif current_ctfidf is not None and prev_ctfidf is not None:
			sim_mat = _matrix(current_ctfidf, prev_ctfidf)
		else:
			return []

		matches: List[Tuple[int, int, float]] = []
		used_prev = set()
		for curr_idx, curr_id in enumerate(curr_ids):
			best_prev = None
			best_sim = -1.0
			for prev_idx, prev_id in enumerate(prev_ids):
				if prev_id in used_prev:
					continue
				sim = sim_mat[curr_idx, prev_idx]
				if sim > best_sim:
					best_sim = sim
					best_prev = prev_id
			if best_prev is not None:
				used_prev.add(best_prev)
				matches.append((curr_id, best_prev, float(best_sim)))
		return matches
