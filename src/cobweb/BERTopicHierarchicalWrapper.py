"""
Wrapper to make CobwebClusterer class ready for BERTopic Library!!

https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.cluster import hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform
from bertopic._utils import select_topic_representation

class BERTopicHierarchicalWrapper:
    def __init__(self, docs, bertopic_model, linkage_function = None, cobweb_clusterer = None, topk = 15):
        self.topk = topk
        if cobweb_clusterer is not None:
            self._from_cobweb_clusterer(docs, cobweb_clusterer, bertopic_model)
        else:
            self._from_bertopic_hierarchical(docs, bertopic_model, linkage_function)
        
    def _from_bertopic_hierarchical(self, docs, bertopic_model, linkage_function):
        # Prepare vocabulary words from the BERTopic vectorizer
        try:
            words = bertopic_model.vectorizer_model.get_feature_names_out()
        except Exception:
            words = bertopic_model.vectorizer_model.get_feature_names()

        # Build per-topic bag-of-words counts (excluding outliers -1)
        documents = pd.DataFrame({
            "Document": docs,
            "ID": range(len(docs)),
            "Topic": bertopic_model.topics_,
        })
        documents_per_topic = documents.groupby(["Topic"], as_index=False).agg({"Document": " ".join})
        documents_per_topic = documents_per_topic.loc[documents_per_topic.Topic != -1, :]

        clean_documents = bertopic_model._preprocess_text(documents_per_topic.Document.values)
        bow = bertopic_model.vectorizer_model.transform(clean_documents)

        # Topic id order as in bow
        topic_ids = [int(t) for t in documents_per_topic.Topic.values]

        # Compute condensed distance matrix using topic embeddings (not c-TF-IDF)
        embeddings = select_topic_representation(
            bertopic_model.c_tf_idf_,
            bertopic_model.topic_embeddings_,
            use_ctfidf=False,
        )[0]
        # Exclude outlier topics if present
        try:
            outliers = int(getattr(bertopic_model, "_outliers", 0))
        except Exception:
            outliers = 0
        embeddings = embeddings[outliers:]

        sim = cosine_similarity(embeddings)
        np.fill_diagonal(sim, 1.0)
        dist_square = 1.0 - sim
        np.fill_diagonal(dist_square, 0.0)
        dist_square[dist_square < 0] = 0.0
        dist_condensed = squareform(dist_square, checks=False)

        # Linkage
        if linkage_function is None:
            Z = sch.linkage(dist_condensed, method="ward", optimal_ordering=True)
        else:
            Z = linkage_function(dist_condensed)

        # Build adjacency: parent_id = n + i; children map leaves to topic ids
        n = embeddings.shape[0]
        adjacency = {}
        for i in range(len(Z)):
            parent_id = int(n + i)
            left_raw = int(Z[i][0])
            right_raw = int(Z[i][1])
            left_id = topic_ids[left_raw] if left_raw < n else int(left_raw)
            right_id = topic_ids[right_raw] if right_raw < n else int(right_raw)
            adjacency[parent_id] = [left_id, right_id]

        # Compute node levels (leaf = 0; parent = 1 + max(children))
        leaves = set(topic_ids)
        levels = {int(l): 0 for l in leaves}
        unresolved_parents = set(adjacency.keys())
        while unresolved_parents:
            resolved_now = set()
            for parent in unresolved_parents:
                c1, c2 = adjacency[parent]
                if c1 in levels and c2 in levels:
                    levels[parent] = max(levels[c1], levels[c2]) + 1
                    resolved_now.add(parent)
            if not resolved_now:
                break
            unresolved_parents = unresolved_parents.difference(resolved_now)

        # Map actual BERTopic topic id -> row index in bow
        topic_id_to_row = {int(t): i for i, t in enumerate(documents_per_topic.Topic.values)}

        # Helper: extract top-k frequent words from a sparse count vector
        def topk_words_from_row(row: csr_matrix, k: int):
            counts = row.toarray().ravel()
            if counts.size == 0:
                return []
            # Keep only indices with non-zero counts
            non_zero = np.flatnonzero(counts)
            if non_zero.size == 0:
                return []
            if non_zero.size <= k:
                indices_sorted = non_zero[np.argsort(-counts[non_zero])]
            else:
                top_idx = np.argpartition(counts[non_zero], -k)[-k:]
                indices_sorted = non_zero[top_idx][np.argsort(-counts[non_zero][top_idx])]
            return [str(words[j]) for j in indices_sorted]

        # Compute leaf sets for each node to aggregate counts for parents
        node_to_leaves = {int(l): {int(l)} for l in leaves}
        unresolved = set(adjacency.keys())
        while unresolved:
            progressed = set()
            for parent in unresolved:
                c1, c2 = adjacency[parent]
                if c1 in node_to_leaves and c2 in node_to_leaves:
                    node_to_leaves[parent] = node_to_leaves[c1].union(node_to_leaves[c2])
                    progressed.add(parent)
            if not progressed:
                break
            unresolved = unresolved.difference(progressed)

        # Build rows with required fields: Node_ID, Level, Keywords, children_ids
        rows = []

        # Leaves
        for leaf in sorted(leaves):
            leaf_id = int(leaf)
            keywords = []
            if leaf_id in topic_id_to_row:
                row_idx = topic_id_to_row[leaf_id]
                keywords = topk_words_from_row(bow[row_idx], self.topk)
            rows.append({
                "Node_ID": leaf_id,
                "Level": levels.get(leaf_id, 0),
                "Keywords": keywords,
                "children_ids": [],
            })

        # Parents
        for parent in sorted(adjacency.keys()):
            leaf_ids = node_to_leaves.get(parent, set())
            # Sum bow rows for all leaves under this parent
            indices = [topic_id_to_row[lid] for lid in leaf_ids if lid in topic_id_to_row]
            if indices:
                grouped = csr_matrix(bow[indices].sum(axis=0))
                keywords = topk_words_from_row(grouped, self.topk)
            else:
                keywords = []
            rows.append({
                "Node_ID": int(parent),
                "Level": levels.get(int(parent), None),
                "Keywords": keywords,
                "children_ids": adjacency.get(int(parent), []),
            })

        df = pd.DataFrame(rows).sort_values(["Level", "Node_ID"]).reset_index(drop=True)
        self.hierachical_topics = df


    ## from CobwebClusterer
    def _from_cobweb_clusterer(self, docs, cobweb_clusterer, bertopic_model):
        # Create mapping from nodes to their document indices and children
        doc_map, children_map = cobweb_clusterer._create_node_doc_assignment()

        # Prepare vocabulary words from the BERTopic vectorizer
        try:
            words = bertopic_model.vectorizer_model.get_feature_names_out()
        except Exception:
            words = bertopic_model.vectorizer_model.get_feature_names()

        # Preprocess all docs and build BOW for per-document frequencies
        clean_docs = bertopic_model._preprocess_text(docs)
        bow_docs = bertopic_model.vectorizer_model.transform(clean_docs)

        # Helper: extract top-k frequent words from a sparse count vector
        def topk_words_from_row(row: csr_matrix, k: int):
            counts = row.toarray().ravel()
            if counts.size == 0:
                return []
            non_zero = np.flatnonzero(counts)
            if non_zero.size == 0:
                return []
            if non_zero.size <= k:
                indices_sorted = non_zero[np.argsort(-counts[non_zero])]
            else:
                top_idx = np.argpartition(counts[non_zero], -k)[-k:]
                indices_sorted = non_zero[top_idx][np.argsort(-counts[non_zero][top_idx])]
            return [str(words[j]) for j in indices_sorted]

        # Build adjacency by node id and collect all nodes
        adjacency = {}
        node_ids = set()
        for node, children in children_map.items():
            nid = int(getattr(node, 'id', hash(node)))
            adjacency[nid] = [int(getattr(c, 'id', hash(c))) for c in children]
            node_ids.add(nid)
            for c in children:
                node_ids.add(int(getattr(c, 'id', hash(c))) )

        # Include nodes that may have no children but are present in doc_map
        for node in doc_map.keys():
            nid = int(getattr(node, 'id', hash(node)))
            node_ids.add(nid)
            adjacency.setdefault(nid, [])

        # Determine leaves (nodes with no children)
        leaves = {nid for nid in node_ids if len(adjacency.get(nid, [])) == 0}

        # Compute levels: leaves = 0; parents = 1 + max(children)
        levels = {nid: 0 for nid in leaves}
        unresolved_parents = {nid for nid in node_ids if len(adjacency.get(nid, [])) > 0}
        # Map id -> node object for doc lookup
        id_to_node = {int(getattr(node, 'id', hash(node))): node for node in doc_map.keys()}

        while unresolved_parents:
            resolved_now = set()
            for parent in list(unresolved_parents):
                children = adjacency[parent]
                if all((c in levels) for c in children):
                    levels[parent] = (max(levels[c] for c in children) + 1) if children else 0
                    resolved_now.add(parent)
            if not resolved_now:
                break
            unresolved_parents = unresolved_parents.difference(resolved_now)

        # Aggregate BOW per node using document indices from cobweb clusterer
        rows = []
        for nid in sorted(node_ids, key=lambda x: (levels.get(x, 0), x)):
            node_obj = id_to_node.get(nid, None)
            doc_indices = []
            if node_obj is not None:
                doc_indices = doc_map.get(node_obj, []) or []
            # Sum BOW for all doc indices under this node
            if len(doc_indices) > 0:
                grouped = csr_matrix(bow_docs[doc_indices].sum(axis=0))
                keywords = topk_words_from_row(grouped, self.topk)
            else:
                keywords = []
            rows.append({
                "Node_ID": nid,
                "Level": levels.get(nid, 0),
                "Keywords": keywords,
                "children_ids": adjacency.get(nid, []),
            })

        df = pd.DataFrame(rows).sort_values(["Level", "Node_ID"]).reset_index(drop=True)
        self.hierachical_topics = df
        