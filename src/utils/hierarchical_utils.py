"""
Hierarchical BERTopic runner and utilities.

This mirrors the structure of the non-hierarchical runner in `bertopic_utils.py`
but evaluates hierarchical metrics using `BERTopicHierarchicalWrapper` to build
the hierarchy and the metrics implemented in `Cobweb-TopicModeling/src/benchmarks/evaluation.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from bertopic import BERTopic

from src.utils.bertopic_utils import BERTopicDataset
from src.cobweb.BERTopicHierarchicalWrapper import BERTopicHierarchicalWrapper

logger = logging.getLogger(__name__)


#############################
# Local hierarchical metrics #
#############################

def compute_npmi(doc_word: np.ndarray, topic_word: np.ndarray, n_list: List[int]) -> float:
	"""Compute NPMI coherence using binary doc-word and topic distributions."""
	topic_size, word_size = np.shape(topic_word)
	doc_size = np.shape(doc_word)[0]
	if topic_size == 0 or doc_size == 0 or word_size == 0:
		return 0.0
	scores = []
	for N in n_list:
		# top-N words per topic
		top_idxs = [np.argpartition(topic_word[t, :], -min(N, word_size))[-min(N, word_size):] for t in range(topic_size)]
		level_sum = 0.0
		for idxs in top_idxs:
			if len(idxs) < 2:
				continue
			topic_sum = 0.0
			pairs = 0
			for i in range(len(idxs)):
				wi = idxs[i]
				fi = doc_word[:, wi] > 0
				p_i = float(fi.sum()) / doc_size
				for j in range(i + 1, len(idxs)):
					wj = idxs[j]
					fj = doc_word[:, wj] > 0
					p_j = float(fj.sum()) / doc_size
					p_ij = float((fi & fj).sum()) / doc_size
					if p_ij > 0 and p_i > 0 and p_j > 0:
						topic_sum += np.log(p_ij / (p_i * p_j)) / (-np.log(p_ij))
						pairs += 1
			if pairs > 0:
				# average over pairs within topic
				topic_sum *= (2.0 / (len(idxs) * (len(idxs) - 1)))
				level_sum += topic_sum
		scores.append(level_sum / max(topic_size, 1))
	return float(np.mean(scores)) if scores else 0.0


def compute_topic_uniqueness(topic_word: np.ndarray, n: int) -> float:
	"""Compute Topic Uniqueness: average inverse frequency of top words across topics."""
	T, V = topic_word.shape if topic_word.ndim == 2 else (0, 0)
	if T == 0 or V == 0:
		return 0.0
	top_lists = [np.argpartition(topic_word[t], -min(n, V))[-min(n, V):] for t in range(T)]
	counts = np.zeros(V, dtype=np.int32)
	for lst in top_lists:
		counts[lst] += 1
	tu_total = 0.0
	for lst in top_lists:
		inv_sum = 0.0
		for w in lst:
			c = counts[w]
			inv_sum += (1.0 / c) if c > 0 else 0.0
		tu_total += inv_sum / max(len(lst), 1)
	return float(tu_total / T)


def evaluate_TU(topic_word: np.ndarray, n_list: List[int]) -> float:
	return float(np.mean([compute_topic_uniqueness(topic_word, n) for n in n_list]))


def compute_TQ(doc_word: np.ndarray, topic_word: np.ndarray, n_list: List[int]) -> float:
	tu = evaluate_TU(topic_word, n_list)
	npmi = compute_npmi(doc_word, topic_word, n_list)
	return float(tu * npmi)


def compute_topic_diversity(topic_words: List[List[str]]) -> float:
	if not topic_words:
		return 0.0
	flat = sum(topic_words, [])
	if not flat:
		return 0.0
	return float(len(set(flat)) / len(flat))


def compute_topic_pair_difference(words_a: List[str], words_b: List[str]) -> float:
	"""Compute difference ratio between two topics' word lists.

	Matches TraCo's pair difference: fraction of words that appear in
	only one of the two topics, divided by the combined length.
	"""
	if not words_a and not words_b:
		return 0.0
	from collections import Counter
	c = Counter()
	c.update(words_a)
	c.update(words_b)
	unique_only = sum(1 for v in c.values() if v == 1)
	denom = len(words_a) + len(words_b)
	return float(unique_only / denom) if denom > 0 else 0.0


def compute_group_td(groups: List[List[List[str]]]) -> float:
	"""Compute average topic diversity across sibling groups.

	Each group is a list of topics (each as a list of words). For a group,
	count words across topics and compute the fraction that occur exactly once
	divided by total tokens, then average across groups.
	"""
	if not groups:
		return 0.0
	group_scores: List[float] = []
	from collections import Counter
	for topics in groups:
		flat = []
		for tw in topics:
			flat.extend(tw)
		if not flat:
			group_scores.append(0.0)
			continue
		cnt = Counter(flat)
		unique_once = sum(1 for v in cnt.values() if v == 1)
		group_scores.append(float(unique_once / len(flat)))
	return float(np.mean(group_scores)) if group_scores else 0.0


def compute_topic_specialization(topic_word: np.ndarray, doc_word: np.ndarray) -> float:
	"""Compute specialization as 1 - cosine similarity to corpus vector."""
	if topic_word.size == 0 or doc_word.size == 0:
		return 0.0
	corpus_vec = doc_word.sum(axis=0).astype(np.float64)
	cnorm = np.linalg.norm(corpus_vec)
	if cnorm == 0:
		return 0.0
	corpus_vec = corpus_vec / cnorm
	tw = topic_word.astype(np.float64)
	norms = np.linalg.norm(tw, axis=1)
	norms[norms == 0] = 1.0
	tw_norm = (tw.T / norms).T
	sims = tw_norm.dot(corpus_vec)
	return float(np.mean(1.0 - sims))


def compute_hierarchical_affinity(relations: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
	"""Affinity: child similarity to closest parent vs. others."""
	if not relations:
		return 0.0, 0.0
	child_scores = []
	non_child_scores = []
	for child_dist, parent_dist in relations:
		if child_dist.size == 0 or parent_dist.size == 0:
			continue
		c_norm = np.linalg.norm(child_dist, axis=1, keepdims=True)
		p_norm = np.linalg.norm(parent_dist, axis=1, keepdims=True)
		c_norm[c_norm == 0] = 1.0
		p_norm[p_norm == 0] = 1.0
		c_n = child_dist / c_norm
		p_n = parent_dist / p_norm
		sim = c_n @ p_n.T
		max_idx = np.argmax(sim, axis=1)
		for i in range(sim.shape[0]):
			best = sim[i, max_idx[i]]
			child_scores.append(best)
			others = np.delete(sim[i], max_idx[i])
			if others.size:
				non_child_scores.extend(list(others))
	child_aff = float(np.mean(child_scores)) if child_scores else 0.0
	non_child_aff = float(np.mean(non_child_scores)) if non_child_scores else 0.0
	return child_aff, non_child_aff


def compute_clnpmi(child_topic: np.ndarray, parent_topic: np.ndarray, doc_word: np.ndarray) -> float:
	"""Compute CLNPMI.

	Accepts either single topic distributions (1D) or matrices (2D). For 2D
	inputs, pairs each child topic with its most similar parent topic and
	averages the per-pair CLNPMI.
	"""
	if child_topic.size == 0 or parent_topic.size == 0 or doc_word.size == 0:
		return 0.0

	def _clnpmi_pair(c_vec: np.ndarray, p_vec: np.ndarray) -> float:
		doc_n = doc_word.shape[0]
		vals = []
		for N in [5, 10, 15]:
			k_c = min(N, c_vec.size)
			k_p = min(N, p_vec.size)
			idx_c_full = np.argpartition(c_vec, -k_c)[-k_c:]
			idx_p_full = np.argpartition(p_vec, -k_p)[-k_p:]
			# exclude overlaps
			set_c = set([int(x) for x in idx_c_full.tolist()])
			set_p = set([int(x) for x in idx_p_full.tolist()])
			idx_c = [w for w in set_c if w not in set_p]
			idx_p = [w for w in set_p if w not in set_c]
			if not idx_c or not idx_p:
				vals.append(0.0)
				continue
			sum_score = 0.0
			pairs = 0
			for wi in idx_c:
				fi = doc_word[:, wi] > 0
				p_i = float(fi.sum()) / doc_n
				for wj in idx_p:
					fj = doc_word[:, wj] > 0
					p_j = float(fj.sum()) / doc_n
					p_ij = float((fi & fj).sum()) / doc_n
					if p_ij == 1.0:
						sum_score += 1.0
						pairs += 1
					elif p_ij > 0 and p_i > 0 and p_j > 0:
						p_ij = p_ij + 1e-10
						sum_score += np.log(p_ij / (p_i * p_j)) / (-np.log(p_ij))
						pairs += 1
			vals.append(sum_score / pairs if pairs > 0 else 0.0)
		return float(np.mean(vals)) if vals else 0.0

	# Normalize helpers
	def _normalize_rows(mat: np.ndarray) -> np.ndarray:
		norms = np.linalg.norm(mat, axis=1, keepdims=True)
		norms[norms == 0] = 1.0
		return mat / norms

	# Handle 1D cases directly
	if child_topic.ndim == 1 and parent_topic.ndim == 1:
		return _clnpmi_pair(child_topic, parent_topic)

	# Promote 1D to 2D when mismatched
	if child_topic.ndim == 1 and parent_topic.ndim == 2:
		child_topic = child_topic[np.newaxis, :]
	if child_topic.ndim == 2 and parent_topic.ndim == 1:
		parent_topic = parent_topic[np.newaxis, :]

	# Now both are 2D
	C = child_topic
	P = parent_topic
	if C.shape[1] != P.shape[1]:
		# pad/truncate to shared vocab size
		V = min(C.shape[1], P.shape[1])
		C = C[:, :V]
		P = P[:, :V]

	# Pair each child to closest parent via cosine similarity
	Cn = _normalize_rows(C.astype(np.float64))
	Pn = _normalize_rows(P.astype(np.float64))
	sim = Cn @ Pn.T
	best = np.argmax(sim, axis=1)
	scores = []
	for i in range(C.shape[0]):
		scores.append(_clnpmi_pair(C[i], P[best[i]]))
	return float(np.mean(scores)) if scores else 0.0


def _get_vocabulary(model: BERTopic) -> List[str]:
	try:
		return list(model.vectorizer_model.get_feature_names_out())
	except Exception:
		return list(model.vectorizer_model.get_feature_names())


def _preprocess_and_bow(model: BERTopic, docs: List[str]):
	"""Preprocess documents using the BERTopic model and transform to BoW.

	Returns (binary_doc_word, count_doc_word, vocabulary)
	"""
	clean_docs = model._preprocess_text(docs)
	bow_counts = model.vectorizer_model.transform(clean_docs)
	# Ensure dense arrays for downstream numpy ops
	counts = bow_counts.toarray() if hasattr(bow_counts, "toarray") else np.asarray(bow_counts)
	binary = (counts > 0).astype(np.int32)
	vocab = _get_vocabulary(model)
	return binary, counts, vocab


def _build_leaf_doc_index_map(model: BERTopic, df: pd.DataFrame) -> Dict[int, List[int]]:
	"""Map leaf Node_IDs to document indices.

	For BERTopic hierarchies, leaves are topic IDs and we can use `model.topics_`.
	"""
	leaf_ids = [int(row.Node_ID) for _, row in df.iterrows() if int(row.Level) == 0]
	topic_assignments = getattr(model, "topics_", None)
	if topic_assignments is None:
		return {lid: [] for lid in leaf_ids}
	mapping: Dict[int, List[int]] = {lid: [] for lid in leaf_ids}
	for i, t in enumerate(topic_assignments):
		if t in mapping:
			mapping[int(t)].append(i)
	return mapping


def _build_adjacency(df: pd.DataFrame) -> Dict[int, List[int]]:
	adj: Dict[int, List[int]] = {}
	for _, row in df.iterrows():
		nid = int(row.Node_ID)
		children = [int(c) for c in (row.children_ids or [])]
		adj[nid] = children
	return adj


def _compute_leaf_sets(df: pd.DataFrame) -> Dict[int, set]:
	"""Compute for every node the set of leaf Node_IDs beneath it."""
	adj = _build_adjacency(df)
	leaves = {int(row.Node_ID) for _, row in df.iterrows() if int(row.Level) == 0}
	leaf_sets: Dict[int, set] = {leaf: {leaf} for leaf in leaves}

	# Topologically expand until all parents have leaf sets
	unresolved = {nid for nid, ch in adj.items() if len(ch) > 0}
	while unresolved:
		progressed = set()
		for nid in list(unresolved):
			ch = adj.get(nid, [])
			if ch and all((c in leaf_sets) for c in ch):
				s = set()
				for c in ch:
					s.update(leaf_sets[c])
				leaf_sets[nid] = s
				progressed.add(nid)
		if not progressed:
			break
		unresolved = unresolved.difference(progressed)
	return leaf_sets


def _aggregate_topic_distributions(
	df: pd.DataFrame,
	leaf_sets: Dict[int, set],
	leaf_to_docs: Dict[int, List[int]],
	doc_word_counts: np.ndarray,
) -> Dict[int, np.ndarray]:
	"""Aggregate per-node topic distributions by summing BoW counts of leaf documents."""
	node_dist: Dict[int, np.ndarray] = {}
	for _, row in df.iterrows():
		nid = int(row.Node_ID)
		leaves = list(leaf_sets.get(nid, {nid}))
		doc_indices: List[int] = []
		for lid in leaves:
			doc_indices.extend(leaf_to_docs.get(lid, []))
		if doc_indices:
			# Sum over selected documents
			dist = doc_word_counts[doc_indices].sum(axis=0)
		else:
			dist = np.zeros((doc_word_counts.shape[1],), dtype=np.float32)
		node_dist[nid] = dist.astype(np.float32)
	return node_dist


def _level_nodes(df: pd.DataFrame) -> Dict[int, List[int]]:
	by_level: Dict[int, List[int]] = {}
	for _, row in df.iterrows():
		lvl = int(row.Level)
		nid = int(row.Node_ID)
		by_level.setdefault(lvl, []).append(nid)
	return by_level


def _top_words_from_dist(dist: np.ndarray, vocab: List[str], top_n: int) -> List[str]:
	if dist.size == 0 or top_n <= 0:
		return []
	idx = np.argpartition(dist, -min(top_n, dist.size))[-min(top_n, dist.size):]
	idx = idx[np.argsort(-dist[idx])]
	return [str(vocab[i]) for i in idx]


@dataclass
class HierarchicalMetrics:
	hier_coherence_npmi: float
	hier_topic_uniqueness: float
	hier_topic_diversity: float
	hier_topic_specialization: float
	hier_affinity_child: float
	hier_affinity_non_child: float
	hier_coherence_clnpmi: float


class BERTopicHierarchicalRunner:
	"""
	Run one or more BERTopic instances and compute hierarchical metrics.

	Usage mirrors `BERTopicRunner` but leverages `BERTopicHierarchicalWrapper`
	to produce a hierarchy dataframe and evaluates metrics from the sibling repo.
	"""

	def __init__(self, topic_models: Sequence[BERTopic]):
		if not topic_models:
			raise ValueError("Provide at least one BERTopic instance to BERTopicHierarchicalRunner")
		self.topic_models = list(topic_models)

	def run(
		self,
		dataset: BERTopicDataset,
		top_n_words: int = 10,
		max_level: Optional[int] = None,
	) -> List[Dict[str, float]]:
		results: List[Dict[str, float]] = []

		for i, model in enumerate(self.topic_models):
			# Reuse already-fitted models from the non-hierarchical runner when possible
			topic_assignments = getattr(model, "topics_", None)
			needs_fit = topic_assignments is None or (isinstance(topic_assignments, list) and len(topic_assignments) != dataset.size)
			if needs_fit:
				logger.info(
					"Fitting BERTopic model %s on %d docs (hierarchical)",
					model.hdbscan_model.__class__.__name__,
					dataset.size,
				)
				model.fit(dataset.documents, embeddings=dataset.embeddings)
			else:
				logger.info(
					"Reusing already-fitted BERTopic model %s for hierarchical conversion",
					model.hdbscan_model.__class__.__name__,
				)

			# Build hierarchy via wrapper; pass cobweb clusterer if available
			cobweb_clusterer = getattr(model.hdbscan_model, "clusterer", None)
			wrapper = BERTopicHierarchicalWrapper(
				docs=dataset.documents,
				bertopic_model=model,
				linkage_function=None,
				cobweb_clusterer=cobweb_clusterer,
				topk=top_n_words,
			)

			df = getattr(wrapper, "hierachical_topics", None)
			if df is None or not isinstance(df, pd.DataFrame) or df.empty:
				logger.warning("Hierarchical wrapper produced no topics; skipping metrics")
				results.append({"model": model})
				continue

			# Preprocess and build document-word matrices
			doc_word_binary, doc_word_counts, vocab = _preprocess_and_bow(model, dataset.documents)

			# Leaf sets and doc index mapping
			leaf_sets = _compute_leaf_sets(df)
			leaf_to_docs = _build_leaf_doc_index_map(model, df)
			node_dist = _aggregate_topic_distributions(df, leaf_sets, leaf_to_docs, doc_word_counts)

			# Organize nodes by level and build per-level topic-word matrices
			by_level = _level_nodes(df)
			levels_sorted = sorted(by_level.keys())
			if max_level is not None:
				levels_sorted = [lvl for lvl in levels_sorted if lvl <= max_level]

			level_topic_mats: Dict[int, np.ndarray] = {}
			level_topic_words: Dict[int, List[List[str]]] = {}
			node_top_words: Dict[int, List[str]] = {}
			for lvl in levels_sorted:
				nodes = by_level.get(lvl, [])
				mats = np.stack([node_dist[nid] for nid in nodes], axis=0) if nodes else np.zeros((0, len(vocab)))
				level_topic_mats[lvl] = mats
				level_topic_words[lvl] = [
					_top_words_from_dist(node_dist[nid], vocab, top_n_words) for nid in nodes
				]
				# also record per-node words for adjacency-based metrics
				for idx, nid in enumerate(nodes):
					node_top_words[nid] = level_topic_words[lvl][idx]

			# Compute per-level metrics and average across levels
			npmi_vals = []
			tu_vals = []
			td_vals = []
			spec_vals = []

			for lvl in levels_sorted:
				mats = level_topic_mats[lvl]
				words_list = level_topic_words[lvl]
				if mats.shape[0] == 0:
					continue

				npmi_vals.append(compute_npmi(doc_word_binary, mats, [top_n_words]))
				tu_vals.append(evaluate_TU(mats, [top_n_words]))
				td_vals.append(compute_topic_diversity(words_list))
				spec_vals.append(compute_topic_specialization(mats, doc_word_binary))

			hier_npmi = float(np.mean(npmi_vals)) if npmi_vals else float("nan")
			hier_tu = float(np.mean(tu_vals)) if tu_vals else float("nan")
			hier_td = float(np.mean(td_vals)) if td_vals else float("nan")
			hier_spec = float(np.mean(spec_vals)) if spec_vals else float("nan")

			# Hierarchical affinity and clNPMI across consecutive levels
			relations: List[Tuple[np.ndarray, np.ndarray]] = []
			for i_lvl in range(len(levels_sorted) - 1):
				child_lvl = levels_sorted[i_lvl]
				parent_lvl = levels_sorted[i_lvl + 1]
				child_mat = level_topic_mats.get(child_lvl, np.zeros((0, len(vocab))))
				parent_mat = level_topic_mats.get(parent_lvl, np.zeros((0, len(vocab))))
				if child_mat.shape[0] and parent_mat.shape[0]:
					relations.append((child_mat, parent_mat))

			if relations:
				child_aff, non_child_aff = compute_hierarchical_affinity(relations)
				clnpmi_vals = [compute_clnpmi(c, p, doc_word_binary) for (c, p) in relations]
				hier_child_aff = float(child_aff)
				hier_non_child_aff = float(non_child_aff)
				hier_clnpmi = float(np.mean(clnpmi_vals)) if clnpmi_vals else float("nan")
			else:
				hier_child_aff = float("nan")
				hier_non_child_aff = float("nan")
				hier_clnpmi = float("nan")

			# Additional TraCo-inspired metrics: PC_TD, PnonC_TD, Sibling_TD, Sibling clNPMI
			adj = _build_adjacency(df)
			pc_td_levels: List[float] = []
			pnonc_td_levels: List[float] = []
			sibling_td_levels: List[float] = []
			sibling_clnpmi_levels: List[float] = []

			for i_lvl in range(len(levels_sorted) - 1):
				child_lvl = levels_sorted[i_lvl]
				parent_lvl = levels_sorted[i_lvl + 1]

				child_nodes = by_level.get(child_lvl, [])
				parent_nodes = by_level.get(parent_lvl, [])
				if not child_nodes or not parent_nodes:
					continue

				# Parent-Child Topic Diversity (average over actual edges)
				pc_scores: List[float] = []
				# Parent-non-Child Topic Diversity
				pnonc_scores: List[float] = []
				# Sibling TD and Sibling clNPMI per parent
				sibling_group_scores: List[float] = []
				sibling_group_clnpmi: List[float] = []

				child_set = set(child_nodes)
				for pid in parent_nodes:
					children = [c for c in adj.get(pid, []) if c in child_set]
					if children:
						# PC_TD pairs
						p_words = node_top_words.get(pid, [])
						for cid in children:
							c_words = node_top_words.get(cid, [])
							pc_scores.append(compute_topic_pair_difference(p_words, c_words))

						# Sibling TD for this parent
						group_words = [node_top_words.get(cid, []) for cid in children]
						sibling_group_scores.append(compute_group_td([group_words]))

						# Sibling clNPMI for this parent (pairwise among children)
						pair_vals: List[float] = []
						for i_idx in range(len(children)):
							for j_idx in range(i_idx + 1, len(children)):
								d_i = node_dist.get(children[i_idx], np.zeros((len(vocab),), dtype=np.float32))
								d_j = node_dist.get(children[j_idx], np.zeros((len(vocab),), dtype=np.float32))
								pair_vals.append(compute_clnpmi(d_i, d_j, doc_word_binary))
						if pair_vals:
							sibling_group_clnpmi.append(float(np.mean(pair_vals)))

					# Parent vs non-children at same child level
					non_children = [c for c in child_nodes if c not in children]
					if non_children:
						p_words = node_top_words.get(pid, [])
						for cid in non_children:
							c_words = node_top_words.get(cid, [])
							pnonc_scores.append(compute_topic_pair_difference(p_words, c_words))

				if pc_scores:
					pc_td_levels.append(float(np.mean(pc_scores)))
				if pnonc_scores:
					pnonc_td_levels.append(float(np.mean(pnonc_scores)))
				if sibling_group_scores:
					sibling_td_levels.append(float(np.mean(sibling_group_scores)))
				if sibling_group_clnpmi:
					sibling_clnpmi_levels.append(float(np.mean(sibling_group_clnpmi)))

			hier_pc_td = float(np.mean(pc_td_levels)) if pc_td_levels else float("nan")
			hier_pnonc_td = float(np.mean(pnonc_td_levels)) if pnonc_td_levels else float("nan")
			hier_sibling_td = float(np.mean(sibling_td_levels)) if sibling_td_levels else float("nan")
			hier_sibling_clnpmi = float(np.mean(sibling_clnpmi_levels)) if sibling_clnpmi_levels else float("nan")

			metrics: Dict[str, float] = {
				"hier_coherence_npmi": hier_npmi,
				"hier_topic_uniqueness": hier_tu,
				"hier_topic_diversity": hier_td,
				"hier_topic_specialization": hier_spec,
				"hier_affinity_child": hier_child_aff,
				"hier_affinity_non_child": hier_non_child_aff,
				"hier_coherence_clnpmi": hier_clnpmi,
				"hier_PC_TD": hier_pc_td,
				"hier_PnonC_TD": hier_pnonc_td,
				"hier_sibling_TD": hier_sibling_td,
				"hier_sibling_clnpmi": hier_sibling_clnpmi,
			}
			results.append({"model": model, **metrics})
		return results
