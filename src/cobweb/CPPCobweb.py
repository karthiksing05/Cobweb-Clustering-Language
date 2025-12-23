"""
Python translation of the C++ CobwebContinuousTree/CobwebContinuousNode
implementations with categorization utilities from OldCobwebTorchTree.

The goal is to mirror the behaviour of the continuous Cobweb variant while
keeping the public API used by ApproxCobwebWrapper (ifit, categorize,
categorize_transitions, get_basic_level_nodes, analyze_structure).
"""

import heapq
import json
import math
import random
from collections import deque, defaultdict
from typing import List, Optional, Tuple

import numpy as np

import torch


# Operation identifiers (match the C++ enum semantics).
BEST, NEW, MERGE, SPLIT = "best", "new", "merge", "split"


def _logsumexp(values: torch.Tensor) -> torch.Tensor:
	"""Stable log-sum-exp for 1D tensors."""
	if values.numel() == 0:
		return torch.tensor(float("-inf"), device=values.device)
	max_v = values.max()
	return max_v + torch.log(torch.exp(values - max_v).sum())


def _custom_rand() -> float:
	return random.random()


def _is_close_to_zero(t: torch.Tensor, eps: float = 1e-5) -> bool:
	return torch.all(torch.abs(t) < eps).item()


class CPPCobwebNode:
	def __init__(self, size: int, num_labels: int, device: Optional[str] = None):
		self.tree = None  # type: Optional["CPPCobwebTree"]
		self.parent: Optional["CPPCobwebNode"] = None
		self.children: List["CPPCobwebNode"] = []
		self.label_counts = torch.zeros(num_labels, device=device, dtype=torch.float)
		self.mean = torch.zeros(size, device=device, dtype=torch.float)
		self.sum_sq = torch.zeros(size, device=device, dtype=torch.float)
		self.count: float = 0.0
		self.best_count: float = 0.0
		# optional downstream fields
		self.sentence_id = []
		self.true_emb = None

	def clone_shallow(self) -> "CPPCobwebNode":
		new_node = CPPCobwebNode(self.mean.numel(), self.label_counts.numel(), self.mean.device)
		new_node.tree = self.tree
		new_node.parent = self.parent
		new_node.label_counts = self.label_counts.clone()
		new_node.mean = self.mean.clone()
		new_node.sum_sq = self.sum_sq.clone()
		new_node.count = float(self.count)
		return new_node

	# --- bookkeeping helpers ---
	def update_label_count_size(self):
		"""Ensure label_counts matches tree.num_labels."""
		desired = self.tree.num_labels
		current = self.label_counts.numel()
		if desired == current:
			return
		pad = torch.zeros(desired - current, device=self.label_counts.device, dtype=self.label_counts.dtype)
		self.label_counts = torch.cat([self.label_counts, pad], dim=0)
		for child in self.children:
			child.update_label_count_size()

	# --- stats updates ---
	def increment_counts(self, instance: torch.Tensor, labels: torch.Tensor) -> None:
		self.count += 1.0
		delta = instance - self.mean
		self.mean = self.mean + delta / self.count
		self.sum_sq = self.sum_sq + delta * (instance - self.mean)
		self.label_counts = self.label_counts + labels

	def update_counts_from_node(self, other: "CPPCobwebNode") -> None:
		delta = other.mean - self.mean
		self.sum_sq = self.sum_sq + other.sum_sq + delta * delta * ((self.count * other.count) / (self.count + other.count))
		self.mean = (self.count * self.mean + other.count * other.mean) / (self.count + other.count)
		self.count += other.count
		self.label_counts = self.label_counts + other.label_counts

	def remove_counts_from_node(self, other: "CPPCobwebNode") -> None:
		if other.count <= 0 or other.count > self.count:
			print("ERROR --- counts are off for remove_counts_from_node")
		if other.count == self.count:
			self.count = 0.0
			self.mean.zero_()
			self.sum_sq.zero_()
			self.label_counts.zero_()
			return
		n_c = self.count - other.count
		delta = other.mean - self.mean
		self.mean = (self.count * self.mean - other.count * other.mean) / n_c
		self.sum_sq = self.sum_sq - (other.sum_sq + delta * delta * ((self.count * other.count) / n_c))
		self.sum_sq = torch.clamp(self.sum_sq, min=0.0)
		self.count = n_c
		self.label_counts = self.label_counts - other.label_counts

	# --- probability utilities ---
	def log_prob(self, instance: torch.Tensor, labels: torch.Tensor) -> float:
		if self.count <= 0:
			return float("-inf")
		if self.tree.covar_from == 1:
			var = self.sum_sq / self.count + self.tree.prior_var
		else:
			if self.parent is not None and self.parent.count > 0:
				var = self.parent.sum_sq / self.parent.count + self.tree.prior_var
			else:
				var = self.sum_sq / max(self.count, 1.0) + self.tree.prior_var
		score = -0.5 * (torch.log(var) + math.log(2.0 * math.pi) + (instance - self.mean) ** 2 / var).sum()
		s = self.label_counts.sum()
		if labels.sum() > 0 and s > 0:
			log_plabels = torch.log(self.label_counts + self.tree.alpha) - math.log(s + self.tree.num_labels * self.tree.alpha)
			score = score + (log_plabels * labels).sum()
		return float(score)

	def log_prob_class_given_instance(self, instance: torch.Tensor, labels: torch.Tensor) -> float:
		if self.tree.root.count <= 0:
			return float("-inf")
		return self.log_prob(instance, labels) + math.log(self.count) - math.log(self.tree.root.count)

	def log_prob_children_given_instance(self, instance: torch.Tensor, labels: torch.Tensor) -> List[float]:
		raw = torch.tensor([c.log_prob_class_given_instance(instance, labels) for c in self.children], device=instance.device)
		log_p_x = torch.logsumexp(raw, dim=0)
		return (raw - log_p_x).tolist()

	def predict_plabels(self, instance: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		denom = self.label_counts.sum() + self.tree.num_labels * self.tree.alpha
		if denom == 0:
			return torch.full((self.tree.num_labels,), 1.0 / max(self.tree.num_labels, 1), device=self.label_counts.device)
		return (self.label_counts + self.tree.alpha) / denom

	def entropy(self) -> float:
		if self.count <= 0:
			return 0.0
		var = self.sum_sq / self.count + self.tree.prior_var
		score = 0.5 * (torch.log(var) + math.log(2.0 * math.pi) + 1.0).sum()
		plabels = (self.label_counts + self.tree.alpha) / (self.label_counts.sum() + self.tree.num_labels * self.tree.alpha)
		score = score - (plabels * torch.log(plabels + 1e-12)).sum()
		return float(score)

	def cross_entropy(self, other: "CPPCobwebNode") -> float:
		if other.count <= 0:
			return float("-inf")
		if self.tree.covar_from == 1:
			var = self.sum_sq / self.count + self.tree.prior_var
		else:
			if self.parent is not None and self.parent.count > 0:
				var = self.parent.sum_sq / self.parent.count + self.tree.prior_var
			else:
				var = self.sum_sq / max(self.count, 1.0) + self.tree.prior_var
		var_other = other.sum_sq / other.count + self.tree.prior_var
		score = -0.5 * (torch.log(var) + math.log(2.0 * math.pi) + var_other / var + (other.mean - self.mean) ** 2 / var).sum()

		label_sum = self.label_counts.sum()
		label_sum_other = other.label_counts.sum()
		if label_sum_other > 0 and label_sum > 0:
			log_plabels = torch.log(self.label_counts + self.tree.alpha) - math.log(label_sum + self.tree.num_labels * self.tree.alpha)
			plabels_other = (other.label_counts + self.tree.alpha) / (label_sum_other + self.tree.num_labels * self.tree.alpha)
			score = score + (log_plabels * plabels_other).sum()
		return float(score)

	# --- PU calculations ---
	def mean_var_plabels(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		var = self.tree.compute_var(self.sum_sq, self.count)
		plabels = (self.label_counts + self.tree.alpha) / (self.label_counts.sum() + self.tree.num_labels * self.tree.alpha)
		return self.mean, var, plabels

	def mean_var_plabels_new(self, instance: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		plabels = (labels + self.tree.alpha) / (labels.sum() + self.tree.num_labels * self.tree.alpha)
		var = self.tree.prior_var
		return instance, var, plabels

	def mean_var_plabels_insert(self, instance: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		count = self.count + 1.0
		delta = instance - self.mean
		mean = self.mean + delta / count
		sum_sq = self.sum_sq + delta * (instance - mean)
		var = self.tree.compute_var(sum_sq, count)
		label_counts = self.label_counts + labels
		plabels = (label_counts + self.tree.alpha) / (label_counts.sum() + self.tree.num_labels * self.tree.alpha)
		return mean, var, plabels

	def mean_var_plabels_insert_node(self, other: "CPPCobwebNode") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		delta = other.mean - self.mean
		sum_sq = self.sum_sq + other.sum_sq + delta * delta * ((self.count * other.count) / (self.count + other.count))
		mean = (self.count * self.mean + other.count * other.mean) / (self.count + other.count)
		count = self.count + other.count
		var = self.tree.compute_var(sum_sq, count)
		label_counts = self.label_counts + other.label_counts
		plabels = (label_counts + self.tree.alpha) / (label_counts.sum() + self.tree.num_labels * self.tree.alpha)
		return mean, var, plabels

	def mean_var_plabels_remove(self, other: "CPPCobwebNode") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		n_c = self.count - other.count
		delta = other.mean - self.mean
		mean = (self.count * self.mean - other.count * other.mean) / n_c
		sum_sq = self.sum_sq - (other.sum_sq + delta * delta * ((self.count * other.count) / n_c))
		sum_sq = torch.clamp(sum_sq, min=0.0)
		var = self.tree.compute_var(sum_sq, n_c)
		label_counts = self.label_counts - other.label_counts
		plabels = (label_counts + self.tree.alpha) / (label_counts.sum() + self.tree.num_labels * self.tree.alpha)
		return mean, var, plabels

	def mean_var_plabels_merge(self, other: "CPPCobwebNode", instance: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		delta = other.mean - self.mean
		sum_sq = self.sum_sq + other.sum_sq + delta * delta * ((self.count * other.count) / (self.count + other.count))
		mean = (self.count * self.mean + other.count * other.mean) / (self.count + other.count)
		count = self.count + other.count
		count += 1.0
		delta2 = instance - mean
		mean = mean + delta2 / count
		sum_sq = sum_sq + delta2 * (instance - mean)
		var = self.tree.compute_var(sum_sq, count)
		label_counts = self.label_counts + other.label_counts + labels
		plabels = (label_counts + self.tree.alpha) / (label_counts.sum() + self.tree.num_labels * self.tree.alpha)
		return mean, var, plabels

	def mean_var_plabels_merge_node(self, other: "CPPCobwebNode", node: "CPPCobwebNode") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		delta = other.mean - self.mean
		sum_sq = self.sum_sq + other.sum_sq + delta * delta * ((self.count * other.count) / (self.count + other.count))
		mean = (self.count * self.mean + other.count * other.mean) / (self.count + other.count)
		count = self.count + other.count
		delta2 = node.mean - mean
		sum_sq = sum_sq + node.sum_sq + delta2 * delta2 * ((count * node.count) / (count + node.count))
		mean = (count * mean + node.count * node.mean) / (count + node.count)
		count += node.count
		var = self.tree.compute_var(sum_sq, count)
		label_counts = self.label_counts + other.label_counts + node.label_counts
		plabels = (label_counts + self.tree.alpha) / (label_counts.sum() + self.tree.num_labels * self.tree.alpha)
		return mean, var, plabels

	# --- PU scores for operations ---
	def pu_for_insert(self, child: "CPPCobwebNode", node: "CPPCobwebNode") -> float:
		parent_mean, parent_var, parent_plabels = self.mean_var_plabels_insert_node(node)
		score = 0.0
		for c in self.children:
			if c is child:
				p_child = (c.count + node.count) / (self.count + node.count)
				c_mean, c_var, c_plabel = c.mean_var_plabels_insert_node(node)
			else:
				p_child = c.count / (self.count + node.count)
				c_mean, c_var, c_plabel = c.mean_var_plabels()
			score += p_child * self.tree.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
		return score / max(len(self.children), 1)

	def pu_for_insert_instance(self, child: "CPPCobwebNode", instance: torch.Tensor, labels: torch.Tensor) -> float:
		parent_mean, parent_var, parent_plabels = self.mean_var_plabels_insert(instance, labels)
		score = 0.0
		for c in self.children:
			if c is child:
				p_child = (c.count + 1.0) / (self.count + 1.0)
				c_mean, c_var, c_plabel = c.mean_var_plabels_insert(instance, labels)
			else:
				p_child = c.count / (self.count + 1.0)
				c_mean, c_var, c_plabel = c.mean_var_plabels()
			score += p_child * self.tree.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
		return score / max(len(self.children), 1)

	def pu_for_new_node(self, node: "CPPCobwebNode") -> float:
		parent_mean, parent_var, parent_plabels = self.mean_var_plabels_insert_node(node)
		score = 0.0
		for c in self.children:
			p_child = c.count / (self.count + node.count)
			c_mean, c_var, c_plabel = c.mean_var_plabels()
			score += p_child * self.tree.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
		p_new = node.count / (self.count + node.count)
		c_mean, c_var, c_plabel = node.mean_var_plabels()
		score += p_new * self.tree.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
		return score / (len(self.children) + 1)

	def pu_for_new_instance(self, instance: torch.Tensor, labels: torch.Tensor) -> float:
		parent_mean, parent_var, parent_plabels = self.mean_var_plabels_insert(instance, labels)
		score = 0.0
		for c in self.children:
			p_child = c.count / (self.count + 1.0)
			c_mean, c_var, c_plabel = c.mean_var_plabels()
			score += p_child * self.tree.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
		p_new = 1.0 / (self.count + 1.0)
		c_mean, c_var, c_plabel = self.mean_var_plabels_new(instance, labels)
		score += p_new * self.tree.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
		return score / (len(self.children) + 1)

	def pu_for_merge_node(self, best1: "CPPCobwebNode", best2: "CPPCobwebNode", node: "CPPCobwebNode") -> float:
		parent_mean, parent_var, parent_plabels = self.mean_var_plabels_insert_node(node)
		score = 0.0
		for c in self.children:
			if c in (best1, best2):
				continue
			p_child = c.count / (self.count + node.count)
			c_mean, c_var, c_plabel = c.mean_var_plabels()
			score += p_child * self.tree.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
		p_merge = (best1.count + best2.count + node.count) / (self.count + node.count)
		c_mean, c_var, c_plabel = best1.mean_var_plabels_merge_node(best2, node)
		score += p_merge * self.tree.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
		return score / max(len(self.children) - 1, 1)

	def pu_for_merge_instance(self, best1: "CPPCobwebNode", best2: "CPPCobwebNode", instance: torch.Tensor, labels: torch.Tensor) -> float:
		parent_mean, parent_var, parent_plabels = self.mean_var_plabels_insert(instance, labels)
		score = 0.0
		for c in self.children:
			if c in (best1, best2):
				continue
			p_child = c.count / (self.count + 1.0)
			c_mean, c_var, c_plabel = c.mean_var_plabels()
			score += p_child * self.tree.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
		p_merge = (best1.count + best2.count + 1.0) / (self.count + 1.0)
		c_mean, c_var, c_plabel = best1.mean_var_plabels_merge(other=best2, instance=instance, labels=labels)
		score += p_merge * self.tree.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
		return score / max(len(self.children) - 1, 1)

	def pu_for_split(self, child: "CPPCobwebNode") -> float:
		parent_mean, parent_var, parent_plabels = self.mean_var_plabels()
		score = 0.0
		for c in self.children:
			if c is child:
				continue
			p_child = c.count / self.count
			c_mean, c_var, c_plabel = c.mean_var_plabels()
			score += p_child * self.tree.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
		for c in child.children:
			p_child = c.count / self.count
			c_mean, c_var, c_plabel = c.mean_var_plabels()
			score += p_child * self.tree.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
		denom = (len(self.children) - 1 + len(child.children))
		return score / max(denom, 1)

	# --- choice helpers ---
	def two_best_children_node(self, node: "CPPCobwebNode") -> Tuple[float, "CPPCobwebNode", Optional["CPPCobwebNode"]]:
		parent_mean, parent_var, parent_plabels = self.mean_var_plabels_insert_node(node)
		relative = []
		for child in self.children:
			p_of_c = (child.count + node.count) / (self.count + node.count)
			c_mean, c_var, c_plabel = child.mean_var_plabels_insert_node(node)
			score_gain = p_of_c * self.tree.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
			p_orig = child.count / (self.count + node.count)
			c_mean2, c_var2, c_plabel2 = child.mean_var_plabels()
			score_gain -= p_orig * self.tree.compute_score(c_mean2, c_var2, c_plabel2, parent_mean, parent_var, parent_plabels)
			relative.append((score_gain, child.count, _custom_rand(), child))
		relative.sort(key=lambda x: (-x[0], -x[1], -x[2]))
		best1 = relative[0][3]
		best1_pu = self.pu_for_insert(best1, node)
		best2 = relative[1][3] if len(relative) > 1 else None
		return best1_pu, best1, best2

	def two_best_children_instance(self, instance: torch.Tensor, labels: torch.Tensor) -> Tuple[float, "CPPCobwebNode", Optional["CPPCobwebNode"]]:
		parent_mean, parent_var, parent_plabels = self.mean_var_plabels_insert(instance, labels)
		relative = []
		for child in self.children:
			p_of_c = (child.count + 1.0) / (self.count + 1.0)
			c_mean, c_var, c_plabel = child.mean_var_plabels_insert(instance, labels)
			score_gain = p_of_c * self.tree.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
			p_orig = child.count / (self.count + 1.0)
			c_mean2, c_var2, c_plabel2 = child.mean_var_plabels()
			score_gain -= p_orig * self.tree.compute_score(c_mean2, c_var2, c_plabel2, parent_mean, parent_var, parent_plabels)
			relative.append((score_gain, child.count, _custom_rand(), child))
		relative.sort(key=lambda x: (-x[0], -x[1], -x[2]))
		best1 = relative[0][3]
		best1_pu = self.pu_for_insert_instance(best1, instance, labels)
		best2 = relative[1][3] if len(relative) > 1 else None
		return best1_pu, best1, best2

	def get_best_operation_node(self, node: "CPPCobwebNode", best1: "CPPCobwebNode", best2: Optional["CPPCobwebNode"], best1_pu: float) -> Tuple[float, str]:
		operations = [(best1_pu, _custom_rand(), BEST)]
		if not self.tree.insert_only and len(self.children) < self.tree.branch_max:
			operations.append((self.pu_for_new_node(node), _custom_rand(), NEW))
			if len(self.children) > 2 and best2 is not None:
				operations.append((self.pu_for_merge_node(best1, best2, node), _custom_rand(), MERGE))
			if len(best1.children) > 0 and (len(self.children) - 1 + len(best1.children)) <= self.tree.branch_max:
				operations.append((self.pu_for_split(best1), _custom_rand(), SPLIT))
		operations.sort(key=lambda x: (-x[0], -x[1]))
		best = operations[0]
		return best[0], best[2]

	def get_best_operation_instance(self, instance: torch.Tensor, labels: torch.Tensor, best1: "CPPCobwebNode", best2: Optional["CPPCobwebNode"], best1_pu: float) -> Tuple[float, str]:
		operations = [(best1_pu, _custom_rand(), BEST)]
		if not self.tree.insert_only and len(self.children) < self.tree.branch_max:
			operations.append((self.pu_for_new_instance(instance, labels), _custom_rand(), NEW))
			if len(self.children) > 2 and best2 is not None:
				operations.append((self.pu_for_merge_instance(best1, best2, instance, labels), _custom_rand(), MERGE))
			if len(best1.children) > 0 and (len(self.children) - 1 + len(best1.children)) <= self.tree.branch_max:
				operations.append((self.pu_for_split(best1), _custom_rand(), SPLIT))
		operations.sort(key=lambda x: (-x[0], -x[1]))
		best = operations[0]
		return best[0], best[2]

	# --- misc helpers ---
	def depth(self) -> int:
		if self.parent is None:
			return 0
		return 1 + self.parent.depth()

	def is_exact_match_instance(self, instance: torch.Tensor, labels: torch.Tensor) -> bool:
		if self.count <= 0:
			return False
		var = self.sum_sq / max(self.count, 1.0)
		if not _is_close_to_zero(var):
			return False
		delta = instance - self.mean
		if not _is_close_to_zero(delta):
			return False
		if self.tree.num_labels == 0:
			return True
		p_concept, idx_concept = torch.max(self.label_counts / self.count, dim=0)
		if abs(1 - p_concept.item()) > 1e-5:
			return False
		p_instance, idx_instance = torch.max(labels / max(labels.sum(), 1.0), dim=0)
		if abs(1 - p_instance.item()) > 1e-5:
			return False
		return idx_concept.item() == idx_instance.item()

	def is_exact_match_node(self, other: "CPPCobwebNode") -> bool:
		if self.count <= 0 or other.count <= 0:
			return False
		var = self.sum_sq / max(self.count, 1.0)
		other_var = other.sum_sq / max(other.count, 1.0)
		if not _is_close_to_zero(var) or not _is_close_to_zero(other_var):
			return False
		if not _is_close_to_zero(other.mean - self.mean):
			return False
		if self.tree.num_labels == 0:
			return True
		p_concept, idx_concept = torch.max(self.label_counts / self.count, dim=0)
		p_other, idx_other = torch.max(other.label_counts / other.count, dim=0)
		if abs(1 - p_concept.item()) > 1e-5 or abs(1 - p_other.item()) > 1e-5:
			return False
		return idx_concept.item() == idx_other.item()

	def _hash(self) -> int:
		return hash(id(self))

	def __str__(self) -> str:
		return str(self._hash())

	# --- serialization helpers (lightweight Python versions) ---
	def output_json(self) -> str:
		payload = {
			"name": f"Concept{self._hash()}",
			"size": self.count,
			"children": [json.loads(c.output_json()) for c in self.children],
			"counts": {
				"labels": {i: float(self.label_counts[i].item() + self.tree.alpha) for i in range(self.tree.num_labels)}
			},
		}
		return json.dumps(payload)

	def to_map(self) -> dict:
		return {
			"node_id": str(self._hash()),
			"mean": [float(x) for x in self.mean.tolist()],
			"sum_sq": [float(x) for x in self.sum_sq.tolist()],
			"count": float(self.count),
			"logvar": [float(math.log((self.sum_sq[i] / max(self.count, 1.0)).item() + 1e-8)) for i in range(self.sum_sq.numel())],
			"children": [c.to_map() for c in self.children],
		}

	def export_tree_json(self) -> str:
		children_json = [json.loads(c.export_tree_json()) for c in self.children]
		payload = {
			"node_id": str(self._hash()),
			"mean": [float(x) for x in self.mean.tolist()],
			"count": float(self.count),
			"sum_sq": [float(x) for x in self.sum_sq.tolist()],
		}
		if children_json:
			payload["children"] = children_json
		return json.dumps(payload)

	def save_tree_to_file(self, filename: str) -> None:
		with open(filename, "w", encoding="utf-8") as f:
			f.write(self.export_tree_json())


class CPPCobwebTree:
	def __init__(
		self,
		shape,
		device: Optional[str] = None,
		use_info: bool = True,
		acuity_cutoff: bool = False,
		use_kl: bool = True,
		prior_var: Optional[float] = None,
		alpha: float = 1e-8,
		covar_from: int = 1,
		insert_only: bool = False,
		depth_max: int = 10_000,
		branch_max: int = 10_000,
	):
		# The public args (use_info, acuity_cutoff, use_kl) are kept for API
		# compatibility with OldCobwebTorchTree; compute_score follows the C++ rules.
		self.device = device
		self.shape = shape if isinstance(shape, torch.Size) else shape
		self.size = shape[0] if hasattr(shape, "__len__") else int(shape)
		self.num_labels = 0
		self.covar_from = covar_from
		self.insert_only = insert_only
		self.depth_max = depth_max
		self.branch_max = branch_max
		self.alpha = alpha
		self.use_info = use_info
		self.acuity_cutoff = acuity_cutoff
		self.use_kl = use_kl
		if prior_var is None:
			base = 1.0 / (2.0 * math.e * math.pi)
			self.prior_var = torch.full((self.size,), base, device=self.device, dtype=torch.float)
		elif isinstance(prior_var, float):
			self.prior_var = torch.full((self.size,), prior_var, device=self.device, dtype=torch.float)
		elif isinstance(prior_var, np.ndarray):
			self.prior_var = torch.from_numpy(prior_var).to(self.device).float()
		else:
			# assume torch.Tensor or sequence
			self.prior_var = torch.as_tensor(prior_var, device=self.device, dtype=torch.float)
		self.labels = {}
		self.reverse_labels = {}
		self.t_best = 0
		self.nt_best = 0
		self.clear()

	# --- core helpers ---
	def clear(self):
		self.root = CPPCobwebNode(self.size, self.num_labels, device=self.device)
		self.root.tree = self

	def _ensure_label(self, label) -> torch.Tensor:
		if label is None:
			if self.num_labels == 0:
				return torch.zeros(0, device=self.device)
			return torch.zeros(self.num_labels, device=self.device)
		if label not in self.labels:
			idx = len(self.labels)
			self.labels[label] = idx
			self.reverse_labels[idx] = label
			self.num_labels += 1
			self._expand_label_counts()
		vec = torch.zeros(self.num_labels, device=self.device)
		vec[self.labels[label]] = 1.0
		return vec

	def _expand_label_counts(self):
		# BFS to pad all nodes when a new label is added
		queue = [self.root]
		while queue:
			node = queue.pop()
			node.update_label_count_size()
			queue.extend(node.children)

	def compute_var(self, sum_sq: torch.Tensor, count: float) -> torch.Tensor:
		# Mirror OldCobwebTorchTree: allow acuity cutoff else add prior variance.
		if count <= 0:
			return self.prior_var.clone()
		mean_sq = sum_sq / count
		if self.acuity_cutoff:
			return torch.max(torch.clamp(mean_sq, min=0.0), self.prior_var)
		return mean_sq + self.prior_var

	def compute_score(
		self,
		child_mean: torch.Tensor,
		child_var: torch.Tensor,
		child_p_label: torch.Tensor,
		parent_mean: torch.Tensor,
		parent_var: torch.Tensor,
		parent_p_label: torch.Tensor,
	) -> float:
		if self.covar_from == 1:
			score = 0.5 * (torch.log(parent_var) - torch.log(child_var)).sum()
		else:
			score = 0.5 * ((child_mean - parent_mean) ** 2 / parent_var).sum()
		if self.num_labels > 0:
			score -= -(child_p_label * torch.log(child_p_label + 1e-12)).sum()
			score += -(parent_p_label * torch.log(parent_p_label + 1e-12)).sum()
		return float(score)

	# --- fitting ---
	def ifit(self, instance: torch.Tensor, label=None) -> CPPCobwebNode:
		with torch.no_grad():
			labels_vec = self._ensure_label(label)
			instance = instance.to(self.device)
			return self.cobweb(instance, labels_vec)

	def cobweb(self, instance: torch.Tensor, labels_vec: torch.Tensor) -> CPPCobwebNode:
		current = self.root
		depth = 1
		while depth < self.depth_max:
			if len(current.children) == 0 and (current.count == 0 or current.is_exact_match_instance(instance, labels_vec)):
				current.increment_counts(instance, labels_vec)
				break
			elif len(current.children) == 0:
				new_node = current.clone_shallow()
				current.parent = new_node
				new_node.children.append(current)
				if new_node.parent is None:
					self.root = new_node
				else:
					new_node.parent.children = [new_node if c is current else c for c in new_node.parent.children]
				new_node.increment_counts(instance, labels_vec)
				current = CPPCobwebNode(self.size, self.num_labels, device=self.device)
				current.parent = new_node
				current.tree = self
				current.increment_counts(instance, labels_vec)
				new_node.children.append(current)
				break
			else:
				best1_pu, best1, best2 = current.two_best_children_instance(instance, labels_vec)
				_, best_action = current.get_best_operation_instance(instance, labels_vec, best1, best2, best1_pu)
				if best_action == BEST:
					current.increment_counts(instance, labels_vec)
					current = best1
				elif best_action == NEW:
					current.increment_counts(instance, labels_vec)
					new_child = CPPCobwebNode(self.size, self.num_labels, device=self.device)
					new_child.parent = current
					new_child.tree = self
					new_child.increment_counts(instance, labels_vec)
					current.children.append(new_child)
					current = new_child
					break
				elif best_action == MERGE:
					current.increment_counts(instance, labels_vec)
					new_child = CPPCobwebNode(self.size, self.num_labels, device=self.device)
					new_child.parent = current
					new_child.tree = self
					new_child.update_counts_from_node(best1)
					new_child.update_counts_from_node(best2)
					best1.parent = new_child
					best2.parent = new_child
					new_child.children.extend([best1, best2])
					current.children = [c for c in current.children if c not in (best1, best2)]
					current.children.append(new_child)
					current = new_child
				elif best_action == SPLIT:
					current.children.remove(best1)
					for c in best1.children:
						c.parent = current
						c.tree = self
						current.children.append(c)
					best1.children = []
				else:
					raise ValueError(f"Unknown action {best_action}")
			depth += 1
		return current

	# --- node insertion/removal/redistribution ---
	def insert_node(self, node: CPPCobwebNode) -> None:
		current = self.root
		depth = 1
		if self.root.count == 0:
			self.root = node
			node.tree = self
			node.parent = None
			return
		while depth < self.depth_max:
			if len(current.children) == 0 and current.is_exact_match_node(node):
				current.update_counts_from_node(node)
				return
			elif len(current.children) == 0:
				new_node = current.clone_shallow()
				current.parent = new_node
				new_node.children.append(current)
				if new_node.parent is None:
					self.root = new_node
				else:
					new_node.parent.children = [new_node if c is current else c for c in new_node.parent.children]
				new_node.update_counts_from_node(node)
				node.parent = new_node
				node.tree = self
				new_node.children.append(node)
				return
			else:
				best1_pu, best1, best2 = current.two_best_children_node(node)
				_, best_action = current.get_best_operation_node(node, best1, best2, best1_pu)
				if best_action == BEST:
					current.update_counts_from_node(node)
					current = best1
				elif best_action == NEW:
					current.update_counts_from_node(node)
					node.parent = current
					node.tree = self
					current.children.append(node)
					return
				elif best_action == MERGE:
					current.update_counts_from_node(node)
					new_child = CPPCobwebNode(self.size, self.num_labels, device=self.device)
					new_child.parent = current
					new_child.tree = self
					new_child.update_counts_from_node(best1)
					new_child.update_counts_from_node(best2)
					best1.parent = new_child
					best2.parent = new_child
					new_child.children.extend([best1, best2])
					current.children = [c for c in current.children if c not in (best1, best2)]
					current.children.append(new_child)
					current = new_child
				elif best_action == SPLIT:
					current.children.remove(best1)
					for c in best1.children:
						c.parent = current
						c.tree = self
						current.children.append(c)
					best1.children = []
				else:
					raise ValueError(f"Unknown action {best_action}")
			depth += 1

	def remove_node(self, node: CPPCobwebNode) -> None:
		if node is self.root:
			self.root = CPPCobwebNode(self.size, self.num_labels, device=self.device)
			self.root.tree = self
			return
		current = node.parent
		if current is None:
			return
		if node in current.children:
			current.children.remove(node)
		node.parent = None
		# subtract the removed node's statistics from all ancestors before restructuring
		ancestor = current
		while ancestor is not None:
			ancestor.remove_counts_from_node(node)
			ancestor = ancestor.parent
		while True:
			if len(current.children) == 1 and current is self.root:
				old_root = self.root
				self.root = old_root.children[0]
				self.root.parent = None
				old_root.children = []
				return
			elif len(current.children) <= 1:
				parent = current.parent
				if parent is None:
					break
				parent.children.remove(current)
				for c in current.children:
					c.parent = parent
					parent.children.append(c)
				current = parent
			else:
				if current.parent is None:
					break
				current = current.parent

	def redistribute_node(self) -> bool:
		node = self.sample_misplaced()
		if node is not None:
			self.remove_node(node)
			self.insert_node(node)
			return True
		return False

	def redistribute(self, n: int) -> None:
		redist = 0
		for i in range(n):
			if self.redistribute_node():
				redist += 1
			if (i % 10000 == 0 or i == n - 1) and i != 0:
				print(f"redistributed {redist} of last {max(10_000, n - 1)} samples ({n - i} redistributes to go).")
				redist = 0

	def misplaced_pu(self, node: CPPCobwebNode, ancestor: CPPCobwebNode) -> bool:
		if ancestor.parent is None:
			return False
		if ancestor is node.parent:
			if len(ancestor.children) == 2:
				return False
			return False
		parent_mean, parent_var, parent_plabels = ancestor.parent.mean_var_plabels()
		relative = []
		for child in ancestor.parent.children:
			if child is node:
				p_c = child.count / max(ancestor.parent.count, 1e-8)
				c_mean, c_var, c_plabel = child.mean_var_plabels()
				score_gain = p_c * self.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
				p_c2 = (child.count - node.count) / max(ancestor.parent.count, 1e-8)
				c_mean2, c_var2, c_plabel2 = child.mean_var_plabels_remove(node)
				score_gain -= p_c2 * self.compute_score(c_mean2, c_var2, c_plabel2, parent_mean, parent_var, parent_plabels)
				relative.append((score_gain, child.count - node.count, _custom_rand(), child))
			else:
				p_c = (child.count + node.count) / max(ancestor.parent.count, 1e-8)
				c_mean, c_var, c_plabel = child.mean_var_plabels_insert_node(node)
				score_gain = p_c * self.compute_score(c_mean, c_var, c_plabel, parent_mean, parent_var, parent_plabels)
				p_c2 = child.count / max(ancestor.parent.count, 1e-8)
				c_mean2, c_var2, c_plabel2 = child.mean_var_plabels()
				score_gain -= p_c2 * self.compute_score(c_mean2, c_var2, c_plabel2, parent_mean, parent_var, parent_plabels)
				relative.append((score_gain, child.count, _custom_rand(), child))
		relative.sort(key=lambda x: (-x[0], -x[1], -x[2]))
		best1 = relative[0][3]
		return best1 is not ancestor

	def misplaced_cross_entropy(self, node: CPPCobwebNode, ancestor: CPPCobwebNode) -> bool:
		if ancestor.parent is None:
			return False
		ancestor_ll = ancestor.cross_entropy(node)
		for alt in ancestor.parent.children:
			if alt is ancestor:
				continue
			if alt.cross_entropy(node) > ancestor_ll:
				return True
		return False

	def sample_misplaced(self) -> Optional[CPPCobwebNode]:
		current = self.root
		while True:
			if len(current.children) == 0:
				break
			norm_const = 0.0
			weights = []
			for child in current.children:
				p = child.count / max(current.count, 1e-8)
				e = child.entropy()
				w = p * e
				weights.append((child, w))
				norm_const += w
			if not weights or norm_const <= 0.0:
				break
			r = _custom_rand() * norm_const
			cum = 0.0
			for child, w in weights:
				cum += w
				if r <= cum:
					current = child
					break
			ancestor = current.parent
			while ancestor is not None and ancestor.parent is not None:
				if self.misplaced_pu(current, ancestor):
					return current
				ancestor = ancestor.parent
		return None

	def sample_leaf(self) -> CPPCobwebNode:
		current = self.root
		while True:
			if len(current.children) == 0:
				return current
			r = _custom_rand()
			v = 0.0
			for child in current.children:
				p = child.count / max(current.count, 1e-8)
				if r <= v + p:
					current = child
					break
				v += p
			if _custom_rand() <= 0.1:
				return current

	def get_leaf(self, instance: torch.Tensor, labels_vec: torch.Tensor) -> CPPCobwebNode:
		current = self.root
		while True:
			if len(current.children) == 0:
				return current
			parent = current
			current = None
			best_log_prob = None
			for child in parent.children:
				log_prob = child.log_prob_class_given_instance(instance, labels_vec)
				if current is None or (best_log_prob is None or log_prob > best_log_prob):
					current = child
					best_log_prob = log_prob

	# --- prediction variants ---
	def predict_pmi(self, instance: torch.Tensor, labels_vec: torch.Tensor, max_nodes: int, greedy: bool) -> torch.Tensor:
		p_labels_given_instance = self.predict(instance, labels_vec, max_nodes, False)
		total_weight = None
		labels_out = torch.zeros(self.num_labels, device=self.device)
		nodes_expanded = 0
		root_ll_inst = self.root.log_prob(instance, labels_vec)
		root_plabels = self.root.predict_plabels(instance, labels_vec)
		queue = [(
			root_ll_inst + (root_plabels * torch.log(root_plabels + 1e-12)).sum().item() + (root_plabels * torch.log(p_labels_given_instance + 1e-12)).sum().item(),
			self.root,
		)]
		while queue:
			queue.sort(key=lambda x: x[0], reverse=True)
			curr_score, curr = queue.pop(0)
			nodes_expanded += 1
			if greedy:
				queue = []
			if total_weight is None:
				total_weight = curr_score
			else:
				total_weight = float(torch.log(torch.tensor([math.exp(total_weight) + math.exp(curr_score)])).item())
			curr_plabels = curr.predict_plabels(instance, labels_vec)
			labels_out = labels_out + math.exp(curr_score - total_weight) * (curr_plabels - labels_out)
			if nodes_expanded >= max_nodes:
				break
			for child in curr.children:
				child_ll = child.log_prob(instance, labels_vec)
				child_plabels = child.predict_plabels(instance, labels_vec)
				queue.append((
					child_ll + (child_plabels * torch.log(child_plabels + 1e-12)).sum().item() + (child_plabels * torch.log(p_labels_given_instance + 1e-12)).sum().item(),
					child,
				))
		return labels_out

	def log_prob(self, instance: torch.Tensor, labels_vec: torch.Tensor, max_nodes: int, greedy: bool) -> float:
		total_weight = None
		out = 0.0
		nodes_expanded = 0
		root_ll_inst = self.root.log_prob(instance, labels_vec)
		queue = [(root_ll_inst, 0.0, self.root)]
		while queue:
			queue.sort(key=lambda x: x[0], reverse=True)
			curr_score, curr_ll, curr = queue.pop(0)
			nodes_expanded += 1
			if greedy:
				queue = []
			if total_weight is None:
				total_weight = curr_score
			else:
				total_weight = float(torch.log(torch.tensor([math.exp(total_weight) + math.exp(curr_score)])).item())
			curr_predicted_log_prob = curr.log_prob(instance, labels_vec)
			out = out + math.exp(curr_score - total_weight) * (curr_predicted_log_prob - out)
			if nodes_expanded >= max_nodes:
				break
			log_children_probs = curr.log_prob_children_given_instance(instance, labels_vec)
			for idx, child in enumerate(curr.children):
				child_ll_inst = child.log_prob(instance, labels_vec)
				child_ll_given_parent = log_children_probs[idx]
				child_ll = child_ll_given_parent + curr_ll
				queue.append((child_ll_inst, child_ll, child))
		return float(out)

	# --- prediction helpers (C++ style) ---
	def predict(self, instance: torch.Tensor, labels_vec: torch.Tensor, max_nodes: int = 1_000, greedy: bool = False) -> torch.Tensor:
		return self.predict_helper(instance, labels_vec, max_nodes, greedy)

	def predict_helper(self, instance: torch.Tensor, labels_vec: torch.Tensor, max_nodes: int, greedy: bool) -> torch.Tensor:
		total_weight = 0.0
		labels_out = torch.zeros(self.num_labels, device=self.device)
		nodes_expanded = 0
		root_ll = self.root.log_prob(instance, labels_vec)
		queue = [(root_ll, self.root)]
		best_node = self.root
		best_score = -float("inf")
		while queue:
			queue.sort(key=lambda x: x[0], reverse=True)
			curr_score, curr = queue.pop(0)
			nodes_expanded += 1
			if greedy:
				queue = []
			if total_weight == 0.0:
				total_weight = curr_score
			else:
				total_weight = float(torch.log(torch.tensor([math.exp(total_weight) + math.exp(curr_score)])).item())
			if curr_score > best_score:
				best_score = curr_score
				best_node = curr
			curr_plabels = curr.predict_plabels(instance, labels_vec)
			labels_out = labels_out + math.exp(curr_score - total_weight) * (curr_plabels - labels_out)
			if nodes_expanded >= max_nodes:
				break
			for child in curr.children:
				child_ll = child.log_prob(instance, labels_vec)
				queue.append((child_ll, child))
		best_node.best_count += 1
		if len(best_node.children) == 0:
			self.t_best += 1
		else:
			self.nt_best += 1
		return labels_out

	# --- categorization (ported from OldCobwebTorchTree) ---
	def _cobweb_categorize(self, instance: torch.Tensor, label, use_best: bool, greedy: bool, max_nodes: float, retrieve_k: Optional[int], return_visited: bool = False):
		labels_vec = self._ensure_label(label)
		queue = []
		heapq.heappush(queue, (-self.root.log_prob(instance, labels_vec), 0.0, random.random(), self.root))
		# heapq.heappush(queue, (-self.dot_prod(self.root.mean, instance), 0.0, random.random(), self.root))
		nodes_visited = 0
		visited_hash = {}
		best = self.root
		best_score = -float("inf")
		retrieved = []
		while queue:
			if greedy:
				neg_score, neg_curr_ll, _, curr = queue.pop()
			else:
				neg_score, neg_curr_ll, _, curr = heapq.heappop(queue)
			score = -neg_score
			curr_ll = -neg_curr_ll
			nodes_visited += 1
			visited_hash[hash(curr)] = nodes_visited
			curr.update_label_count_size()
			if score > best_score:
				best = curr
				best_score = score
			if nodes_visited >= max_nodes:
				break
			if getattr(curr, "sentence_id", None):
				heapq.heappush(retrieved, (len(retrieved), random.random(), curr))
			if retrieve_k is not None and len(retrieved) == retrieve_k:
				break
			if len(curr.children) > 0:
				ll_children = torch.zeros(len(curr.children), device=instance.device)
				for i, c in enumerate(curr.children):
					log_prob = c.log_prob_class_given_instance(instance, labels_vec)
					ll_children[i] = log_prob + math.log(max(c.count, 1e-8)) - math.log(max(curr.count, 1e-8))
				log_p_x = torch.logsumexp(ll_children, dim=0)
				add = []
				for i, c in enumerate(curr.children):
					child_ll = ll_children[i] - log_p_x + curr_ll
					child_ll_inst = c.log_prob_class_given_instance(instance, labels_vec)
					child_score = child_ll + child_ll_inst
					if greedy:
						add.append((-child_ll_inst, -child_ll, random.random(), c))
						# add.append((-self.dot_prod(c.mean, instance), -child_ll, random.random(), c))
					else:
						heapq.heappush(queue, (-child_ll_inst, -child_ll, random.random(), c))
						# heapq.heappush(queue, (-self.dot_prod(c.mean, instance), -child_ll, random.random(), c))
				if greedy:
					add.sort()
					queue.extend(add[::-1])
		if return_visited:
			if retrieve_k is None:
				return (best if use_best else curr), visited_hash
			return [retrieved[i][-1] for i in range(retrieve_k)], visited_hash
		if retrieve_k is None:
			return best if use_best else curr
		return [retrieved[i][-1] for i in range(retrieve_k)]

	def categorize(self, instance: torch.Tensor, label=None, use_best: bool = True, greedy: bool = False, max_nodes: float = float("inf"), retrieve_k: Optional[int] = None, return_visited: bool = False):
		with torch.no_grad():
			return self._cobweb_categorize(instance.to(self.device), label, use_best, greedy, max_nodes, retrieve_k, return_visited)

	def categorize_transitions(self, instance: torch.Tensor, label=None, transition_depth: int = 4, use_best: bool = True, greedy: bool = False, max_nodes: float = float("inf"), top_k: int = 5):
		labels_vec = self._ensure_label(label)
		queue = []
		heapq.heappush(queue, (-self.root.log_prob(instance, labels_vec), 0.0, random.random(), self.root, 0))
		# heapq.heappush(queue, (-self.dot_prod(self.root.mean, instance), 0.0, random.random(), self.root, 0))
		nodes_visited = 0
		best = self.root
		best_score = -float("inf")
		retrieved = []
		while queue:
			if greedy:
				neg_score, neg_curr_ll, _, curr, depth = queue.pop()
			else:
				neg_score, neg_curr_ll, _, curr, depth = heapq.heappop(queue)
			score = -neg_score
			curr_ll = -neg_curr_ll
			nodes_visited += 1
			if score > best_score:
				best = curr
				best_score = score
			if nodes_visited >= max_nodes:
				break
			if depth == transition_depth:
				retrieved.append((len(retrieved), score, random.random(), curr))
				if len(retrieved) == top_k:
					break
				continue
			if len(curr.children) > 0:
				ll_children = torch.zeros(len(curr.children), device=instance.device)
				for i, c in enumerate(curr.children):
					log_prob = c.log_prob_class_given_instance(instance, labels_vec)
					ll_children[i] = log_prob + math.log(max(c.count, 1e-8)) - math.log(max(curr.count, 1e-8))
				log_p_x = torch.logsumexp(ll_children, dim=0)
				add = []
				for i, c in enumerate(curr.children):
					child_ll = ll_children[i] - log_p_x + curr_ll
					child_ll_inst = c.log_prob_class_given_instance(instance, labels_vec)
					child_score = child_ll + child_ll_inst # THIS IS BROKEN SOMEHOW
					if greedy:
						add.append((-child_ll_inst, -child_ll, random.random(), c, depth + 1))
						# add.append((-self.dot_prod(c.mean, instance), -child_ll, random.random(), c, depth + 1))
					else:
						heapq.heappush(queue, (-child_ll_inst, -child_ll, random.random(), c, depth + 1))
						# heapq.heappush(queue, (-self.dot_prod(c.mean, instance), -child_ll, random.random(), c, depth + 1))
				if greedy:
					add.sort()
					queue.extend(add[::-1])

		retrieved.sort(key=lambda x: x[1], reverse=True)
		return [x[-1] for x in retrieved]
	
	def dot_prod(self, x1, x2):
		return torch.dot(x1, x2)

	def get_basic_level_nodes(self):
		bl_nodes = {}
		q = [self.root]
		while q:
			curr = q.pop()
			if getattr(curr, "sentence_id", None):
				bl = curr
				if hasattr(curr, "get_best"):
					bl = curr.get_best(curr.mean)
				bl_nodes[str(hash(bl))] = bl
			else:
				q.extend(curr.children)
		return list(bl_nodes.values())

	def analyze_structure(self, verbose=True):
		leaf_count = 0
		level_counts = defaultdict(int)
		leaf_counts = defaultdict(int)
		child_hist = defaultdict(int)
		q = deque([(self.root, 0)])
		while q:
			node, lvl = q.popleft()
			level_counts[lvl] += 1
			if len(node.children) == 0:
				leaf_count += 1
				leaf_counts[lvl] += 1
			else:
				child_hist[len(node.children)] += 1
				for c in node.children:
					q.append((c, lvl + 1))
		if verbose:
			print(f"\nTotal number of leaf nodes: {leaf_count}\n")
			print("Number of nodes at each level:")
			for lvl in sorted(level_counts.keys()):
				print(f"  Level {lvl}: {level_counts[lvl]} node(s)")
			print("Number of leaves at each level:")
			for lvl in sorted(leaf_counts.keys()):
				print(f"  Level {lvl}: {leaf_counts[lvl]} lea(f/ves)")
			print("\nParent nodes by number of children:")
			for n_children in sorted(child_hist.keys()):
				print(f" {child_hist[n_children]} parent(s) with {n_children} child(ren)")

		return leaf_count, level_counts, leaf_counts, child_hist

	# --- serialization (flat list like C++) ---
	def dump_json(self, filename: str) -> None:
		nodes = []
		stack = [(self.root, -1)]
		node_id = 0
		while stack:
			node, parent_id = stack.pop()
			nodes.append({
				"id": node_id,
				"parent_id": None if parent_id == -1 else parent_id,
				"count": float(node.count),
				"best_count": float(node.best_count),
				"mean": [float(x) for x in node.mean.tolist()],
				"sum_sq": [float(x) for x in node.sum_sq.tolist()],
				"label_counts": [float(x) for x in node.label_counts.tolist()],
			})
			current_id = node_id
			node_id += 1
			for child in reversed(node.children):
				stack.append((child, current_id))
		prior_val = self.prior_var.tolist()
		if isinstance(prior_val, list) and len(prior_val) == 1:
			prior_val = prior_val[0]
		payload = {
			"tree_parameters": {
				"size": self.size,
				"num_labels": self.num_labels,
				"covar_from": self.covar_from,
				"alpha": self.alpha,
				"insert_only": self.insert_only,
				"depth_max": self.depth_max,
				"branch_max": self.branch_max,
				"prior_var": prior_val,
			},
			"nodes": nodes,
		}
		with open(filename, "w", encoding="utf-8") as f:
			json.dump(payload, f)

	def load_json(self, filename: str) -> None:
		with open(filename, "r", encoding="utf-8") as f:
			data = json.load(f)
		params = data.get("tree_parameters", {})
		self.size = params.get("size", self.size)
		self.num_labels = params.get("num_labels", self.num_labels)
		self.covar_from = params.get("covar_from", self.covar_from)
		self.alpha = params.get("alpha", self.alpha)
		self.insert_only = params.get("insert_only", self.insert_only)
		self.depth_max = params.get("depth_max", self.depth_max)
		self.branch_max = params.get("branch_max", self.branch_max)
		prior_val = params.get("prior_var", self.prior_var)
		if prior_val is None:
			base = 1.0 / (2.0 * math.e * math.pi)
			self.prior_var = torch.full((self.size,), base, device=self.device, dtype=torch.float)
		elif isinstance(prior_val, float):
			self.prior_var = torch.full((self.size,), prior_val, device=self.device, dtype=torch.float)
		elif isinstance(prior_val, list):
			self.prior_var = torch.tensor(prior_val, device=self.device, dtype=torch.float)
		elif isinstance(prior_val, np.ndarray):
			self.prior_var = torch.from_numpy(prior_val).to(self.device).float()
		else:
			self.prior_var = torch.as_tensor(prior_val, device=self.device, dtype=torch.float)
		id_to_node = {}
		for entry in data.get("nodes", []):
			n = CPPCobwebNode(self.size, self.num_labels, device=self.device)
			n.tree = self
			n.count = float(entry.get("count", 0.0))
			n.best_count = float(entry.get("best_count", 0.0))
			n.mean = torch.tensor(entry.get("mean", []), device=self.device, dtype=torch.float)
			n.sum_sq = torch.tensor(entry.get("sum_sq", []), device=self.device, dtype=torch.float)
			n.label_counts = torch.tensor(entry.get("label_counts", []), device=self.device, dtype=torch.float)
			id_to_node[entry["id"]] = (n, entry.get("parent_id", None))
		self.root = None
		for node_id, (node, parent_id) in id_to_node.items():
			if parent_id is None:
				self.root = node
				continue
			parent = id_to_node[parent_id][0]
			node.parent = parent
			parent.children.append(node)
		if self.root is None and id_to_node:
			self.root = list(id_to_node.values())[0][0]


__all__ = ["CPPCobwebTree", "CPPCobwebNode"]
