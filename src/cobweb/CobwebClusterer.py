import torch
import json
import random
import math
from tqdm import tqdm
from collections import deque
import numpy as np
import os
import hashlib
from graphviz import Digraph

from src.cobweb.CPPCobweb import CPPCobwebTree

class CobwebClusterer:
    def __init__(
            self,
            transition_depth:int=1,
            prior_var=None,
            corpus=None,
            corpus_embeddings=None, # hierarchize by this
            encode_func=lambda x: x
        ):
        """
        Initializes the Cobweb algorithm for clustering with optional sentences and/or embeddings.

        Important new parameters:
        *   first_method - can be 'bfs' or 'dfs'
        *   second_method - can be 'pathsum' or 'dot'
        *   transition_depth - the depth at which to collect transition nodes

        The goal is that depending on the first or second method, we compute first method
        to organize a depth of nodes by best semantic similarity and then compute second method
        to find the best semantic similarity from just the first node.

        We need to build a specific type of index for the root, as well as the nodes that describe
        our "transition level" (k-depth or basic-level nodes; for now we use k-depth nodes). For
        the root node, we build a search index or define the method to rank the transition level
        nodes, while for each transition level node, we build an index or define the method to
        rank the leaves.
        *   'bfs' - we don't build an index, and run the Cobweb 'categorize' function without
            the greedy argument from that node
        *   'dfs' - we don't build an index, and run the Cobweb 'categorize' function WITH
            the greedy argument from that node
        *   'pathsum' - we build an index similar to how CobwebWrapper's predict-fast method works
        *   'dot' - we build an index to do kNN with the embeddings under the given node directly
            (defined by node.mean for those nodes)
        """

        self.encode_func = encode_func

        self.transition_depth = transition_depth

        self.sentences = []
        self.sentence_to_node = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_init_search = 100000

        # Prediction index caching
        self._cluster_index_valid = False

        # Determine embedding shape
        if corpus_embeddings is not None:
            corpus_embeddings = torch.tensor(corpus_embeddings) if isinstance(corpus_embeddings, list) else corpus_embeddings
            self.embedding_shape = corpus_embeddings.shape[1:]
        elif corpus and len(corpus) > 0:
            sample_emb = self.encode_func([corpus[0]])
            self.embedding_shape = sample_emb.shape[1:]

        self.tree = CPPCobwebTree(
            shape=self.embedding_shape,
            device=self.device,
            use_info=True,
            acuity_cutoff=False, # THIS IS THE SECRET SAUCE!!! changing this to true is what makes this work with language!
            use_kl=True,
            prior_var=prior_var, # OLDCobweb, maybe this is the secret sauce??? setting this to None is super valid
            alpha=1e-8
        )

        if corpus_embeddings is not None:
            if corpus is None:
                corpus = [None] * len(corpus_embeddings)
            self.add_sentences(corpus, corpus_embeddings)
        elif corpus is not None and len(corpus) > 0:
            self.add_sentences(corpus)

    def add_sentences(self, new_sentences, new_vectors=None):
        """
        Adds new sentences and/or embeddings to the Cobweb tree.
        If a sentence is None, it is treated as an embedding-only entry.
        """

        if new_vectors is None:
            new_embeddings = self.encode_func(new_sentences)
        else:
            new_embeddings = new_vectors
            if isinstance(new_embeddings, list):
                new_embeddings = torch.tensor(new_embeddings)
            if new_embeddings.shape[1] != self.tree.shape[0]:
                print(f"[Warning] Provided vector dim {new_embeddings.shape[1]} != tree dim {self.tree.shape[0]}, re-encoding...")
                new_embeddings = self.encode_func(new_sentences)

        start_index = len(self.sentences)

        for i, (sent, emb) in tqdm(enumerate(zip(new_sentences, new_embeddings)),
                                   total=len(new_sentences),
                                   desc="Training CobwebTree"):
            self.sentences.append(sent)
            leaf = self.tree.ifit(torch.tensor(emb, device=self.device))
            if leaf.sentence_id is None:
                leaf.sentence_id = []
            leaf.sentence_id.append(start_index + i)
            self.sentence_to_node[start_index + i] = leaf

        self._cluster_index_valid = False

    def _gather_clusters(self, high_count_thres=5, transition_depth=None):
        """
        Helper function to create and initialize all clusters (and for leaves that do not yet have
        clusters, we can just set them to equal -1. 

        Essentially, every node will be equated to a label of some form, and we return both the
        number of labels as well as the label that each node corresponds to (by hash index). At
        runtime, we can use the data-structures we've assembled in this class to efficiently predict topics.

        At some point, we can change this to be -1 for 
        """

        ### UNCOMMENT FOR REDISTRIBUTING
        # self.tree.redistribute(100_000)

        if self._cluster_index_valid:
            return
        
        if self.transition_depth != -1:
            self.transition_nodes = self.tree.categorize_transitions(
                torch.ones(self.embedding_shape, device=self.device),
                transition_depth=self.transition_depth, top_k=1e9
            )
        elif transition_depth:
            self.transition_nodes = self.tree.categorize_transitions(
                torch.ones(self.embedding_shape, device=self.device),
                transition_depth=transition_depth, top_k=1e9
            )
        else:
            raise Exception("transition_depth not passed in both constructor AND self._gather_clusters!")
    
        print(len(self.transition_nodes))

        C = len(self.transition_nodes)
        D = self.transition_nodes[0].mean.numel()
        K = self.transition_nodes[0].label_counts.numel()

        means = torch.empty((C, D), device=self.device)
        vars_ = torch.empty((C, D), device=self.device)
        counts = torch.empty((C,), device=self.device)
        label_counts = torch.empty((C, K), device=self.device)
        valid = torch.zeros((C,), device=self.device, dtype=torch.bool)
        high_count_mask = torch.zeros((C,), device=self.device, dtype=torch.bool)

        for i, node in enumerate(self.transition_nodes):
            means[i] = node.mean
            label_counts[i] = node.label_counts
            counts[i] = node.count
            valid[i] = node.count > 0
            high_count_mask[i] = node.count > high_count_thres

            if node.count > 0:
                if self.tree.covar_from == 1:
                    vars_[i] = node.sum_sq / node.count + self.tree.prior_var
                else:
                    if node.parent is not None and node.parent.count > 0:
                        vars_[i] = node.parent.sum_sq / node.parent.count + self.tree.prior_var
                    else:
                        vars_[i] = node.sum_sq / max(node.count, 1.0) + self.tree.prior_var
            else:
                vars_[i].fill_(1.0)  # placeholder, will be masked

        self._means = means
        self._vars = vars_
        self._counts = counts
        self._label_counts = label_counts
        self._valid_mask = valid
        self._high_count_mask = high_count_mask # TODO NOT IMPLEMENTED


        ## label training data and create labels training
        training_labels = torch.full(
            (len(self.sentences),),
            -1,
            device=self.device
        )

        for i, tnode in enumerate(self.transition_nodes):
            queue = [tnode]
            while len(queue) > 0:
                curr = queue.pop()

                if len(curr.sentence_id) > 0:
                    training_labels[curr.sentence_id[0]] = i

                for c in curr.children:
                    queue.append(c)

        self._cluster_index_valid = True

        return training_labels

    def predict_clusters(self, X):
        """
        Given a set of instances, predicts the cluster they belong to! Assumes that the passed in
        instances have already been transformed by UMAP.

        Need to batch-transform this!! Returns two tensors, one of all the best tensors and one of
        all the best nodes!
        """

        if not self._cluster_index_valid:
            self._gather_clusters(high_count_thres=5)

        x = X[:, None, :]              # (N, 1, D)
        mu = self._means[None, :, :]      # (1, C, D)
        var = self._vars[None, :, :]      # (1, C, D)

        # Gaussian likelihood
        log_gauss = -0.5 * (
            torch.log(var)
            + math.log(2.0 * math.pi)
            + (x - mu).pow(2) / var
        ).sum(dim=-1)                           # (N, C)

        # Label likelihood
        s = self._label_counts.sum(dim=1, keepdim=True)  # (C, 1)

        log_plabels = (
            torch.log(self._label_counts + self.tree.alpha)
            - torch.log(s + self.tree.num_labels * self.tree.alpha)
        )                                       # (C, K)

        labels = torch.zeros(self.tree.num_labels, device=self.device)

        label_term = labels @ log_plabels.T     # (N, C)

        label_mask = labels.sum(dim=1, keepdim=True) > 0
        label_term = label_term * label_mask

        # Class prior
        log_prior = torch.log(self._counts) - math.log(self.tree.root.count)
        log_prior = log_prior[None, :]          # (1, C)

        # Total
        log_prob = log_gauss + label_term + log_prior

        # Mask invalid nodes
        log_prob[:, ~self._valid_mask] = -float("inf")

        best_node_idxs = log_prob.argmax(dim=1)
        best_scores = log_prob.max(dim=1).values

        # NOTE THIS PRUNES ALL SMALL CLUSTERS
        # check whether best index is in the forbidden mask
        flagged = self._high_count_mask[best_node_idxs]  # (N,) bool

        # replace with -1 if flagged
        best_node_idxs = torch.where(
            flagged,
            torch.full_like(best_node_idxs, -1),
            best_node_idxs
        )

        return best_node_idxs, best_scores

    def print_tree(self):
        """
        Recursively prints the tree structure.
        """
        def _print_node(node, depth=0):
            indent = "  " * depth
            label = f"Sentence ID: {getattr(node, 'sentence_id', 'N/A')}"
            print(f"{indent}- Node ID {node.id} {label}")
            sid = getattr(node, "sentence_id", None)
            if sid and sid[0] < len(self.sentences):
                sentence = self.sentences[sid[0]]
                if sentence is not None:
                    print(f"{indent}    \"{sentence}\"")
                else:
                    print(f"{indent}    [Embedding only]")
            for child in getattr(node, "children", []):
                _print_node(child, depth + 1)

        print("\nCobweb Sentence Clustering Tree:")
        _print_node(self.tree.root)

    def __len__(self):
        """
        Returns the number of sentences in the Cobweb tree.
        """
        return len(self.sentences)

    def visualize_query_subtrees(self, query_embeddings, query_texts=None, directory="query_subtrees", k=6, max_nodes_display=500):
        """
        For each query embedding, find top-`k` leaf nodes using `fast_categorize`,
        compute the minimal subtree that contains all those leaves (union of
        ancestor paths), and render that subtree to `directory` (one file per query).

        Args:
            query_embeddings: iterable of embeddings (list, numpy array, or torch tensor).
            directory: output directory for rendered images.
            k: number of top leaves to retrieve per query (passed to `fast_categorize`).
            max_nodes_display: safety cap on number of nodes to render for a single query.
        """

        self.tree.analyze_structure()

        os.makedirs(directory, exist_ok=True)

        # Allow passing raw query texts instead of precomputed embeddings.
        # If `query_embeddings` is a list of strings, treat them as texts and encode.
        # Otherwise, prefer explicit `query_texts` if provided for labeling.
        q_texts = None
        if isinstance(query_embeddings, list) and len(query_embeddings) > 0 and isinstance(query_embeddings[0], str):
            q_texts = query_embeddings
            q_embs = self.encode_func(q_texts)
            if not torch.is_tensor(q_embs):
                q_embs = torch.tensor(q_embs)
            q_embs = q_embs.to(self.device)
        else:
            # If explicit query_texts provided, keep for labels
            if query_texts is not None:
                q_texts = query_texts

            # Normalize numeric embeddings to torch tensor on device
            if torch.is_tensor(query_embeddings):
                q_embs = query_embeddings.to(self.device)
            else:
                try:
                    q_embs = torch.tensor(query_embeddings, device=self.device)
                except Exception:
                    # Fallback: try encoding if embeddings can't be tensorized
                    if query_texts is not None:
                        q_embs = self.encode_func(query_texts)
                        if not torch.is_tensor(q_embs):
                            q_embs = torch.tensor(q_embs)
                        q_embs = q_embs.to(self.device)
                    else:
                        raise

        def get_sentence_label(sid, max_len=250, wrap=40):
            if sid is not None and sid < len(self.sentences):
                sentence = self.sentences[sid]
                if sentence:
                    needs_ellipsis = len(sentence) > max_len
                    truncated = sentence[:max_len].rstrip()
                    if needs_ellipsis:
                        truncated += "..."
                    # Wrap at word boundaries every ~wrap characters
                    words = truncated.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line) + len(word) + 1 > wrap:
                            lines.append(current_line)
                            current_line = word
                        else:
                            current_line += (" " if current_line else "") + word
                    if current_line:
                        lines.append(current_line)
                    return "\n".join(lines)
            return None

        # Iterate queries
        for qi in range(q_embs.shape[0] if q_embs.ndim > 1 else 1):
            if q_embs.ndim > 1:
                x = q_embs[qi]
            else:
                x = q_embs

            # get top-k leaf nodes via tree categorize
            try:
                top_nodes, visited_hash = self.tree.categorize(
                    x,
                    use_best=False,
                    greedy=False,
                    retrieve_k=k,
                    return_visited=True
                )
            except Exception as e:
                print(f"[Warning] categorize failed for query {qi}: {e}")
                continue

            if not top_nodes:
                print(f"Query {qi}: no leaves retrieved")
                continue

            # Collect the actual leaf node objects
            leaf_nodes = list(top_nodes)

            # For each leaf, collect path of ancestors up to root
            nodes_in_subtree = set()
            parent_map = {}

            for leaf in leaf_nodes:
                curr = leaf
                prev = None
                while curr is not None:
                    nodes_in_subtree.add(curr)
                    # remember parent->child mapping for edges
                    par = getattr(curr, 'parent', None)
                    if par is not None:
                        if par not in parent_map:
                            parent_map[par] = set()
                        parent_map[par].add(curr)
                    prev = curr
                    curr = getattr(curr, 'parent', None)

            if len(nodes_in_subtree) == 0:
                print(f"Query {qi}: empty subtree")
                continue

            if len(nodes_in_subtree) > max_nodes_display:
                print(f"Query {qi}: subtree too large ({len(nodes_in_subtree)} nodes), skipping render")
                continue

            # Render subtree with graphviz
            dot = Digraph(comment=f"Query_{qi}_Subtree", format='png')
            dot.attr(rankdir='TB')
            dot.attr('edge', color='lightblue')

            # Create a small boxed node for the query text in the top corner.
            label_text = None
            if q_texts is not None and qi < len(q_texts):
                label_text = q_texts[qi]
            if label_text is None:
                label_text = f"<embedding_{qi}>"

            # Truncate and wrap label to reasonable length and line width
            def wrap_query_text(text, max_len=200, wrap=40):
                if text is None:
                    return "Query:"
                needs_ellipsis = len(text) > max_len
                truncated = text[:max_len].rstrip()
                if needs_ellipsis:
                    truncated += "..."
                words = truncated.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + (1 if current_line else 0) > wrap:
                        lines.append(current_line)
                        current_line = word
                    else:
                        current_line += (" " if current_line else "") + word
                if current_line:
                    lines.append(current_line)
                if not lines:
                    return "Query:"
                # Put 'Query:' on its own first line so the message wraps below it
                return "Query:\n" + "\n".join(lines)

            wrapped_label = wrap_query_text(label_text, max_len=200, wrap=40)

            # Place the query label inside a tiny top-ranked subgraph so it sits at the top.
            qnode_name = f"q{qi}_label"
            with dot.subgraph(name=f"cluster_q_{qi}") as c:
                c.attr(rank='min')
                c.node(qnode_name, wrapped_label, shape='box', fontsize='12', fontname='Helvetica', style='filled,rounded', fillcolor='lightyellow', margin='0.08,0.05')

            node_ids = {}
            local_counter = {"id": 0}

            def local_next_id():
                local_counter["id"] += 1
                return f"n{local_counter['id']}"

            # Create nodes
            for node in nodes_in_subtree:
                nid = local_next_id()
                node_ids[node] = nid
                sid = getattr(node, 'sentence_id', None)
                # If node is a leaf with sentence, label it; otherwise show depth
                if sid and isinstance(sid, list) and len(sid) and sid[0] is not None and sid[0] < len(self.sentences):
                    label = get_sentence_label(sid[0])
                    if not label:
                        label = "[Embedding only]"
                    label = f"[Count: {visited_hash[hash(node)]}] " + label
                    # Leaf node: emphasize text, larger font and margin
                    dot.node(nid, label, shape='box', style='filled', color='lightgrey', fontsize='16', fontname='Helvetica', margin='0.2,0.1')
                else:
                    # internal node: label with its depth from the root (root=0)
                    # Prefer computing depth relative to `self.tree.root` when available.
                    depth = 0
                    curr_depth_node = node
                    root_node = getattr(self.tree, 'root', None)

                    if root_node is None:
                        # Fall back to parent None termination if tree.root not present
                        while getattr(curr_depth_node, 'parent', None) is not None:
                            depth += 1
                            curr_depth_node = getattr(curr_depth_node, 'parent')
                    else:
                        # Walk up until we reach the actual root object
                        # Guard against malformed trees by also checking for missing parent
                        while curr_depth_node is not root_node and getattr(curr_depth_node, 'parent', None) is not None:
                            depth += 1
                            curr_depth_node = getattr(curr_depth_node, 'parent')

                    depth_label = str(depth) + " : " + str(visited_hash[hash(node)])
                    dot.node(
                        nid,
                        depth_label,
                        shape='oval',
                        width='0.50',
                        height='0.35',
                        fixedsize='true',
                        style='filled',
                        color='#ccccff',
                        fontsize='10',
                        fontname='Helvetica'
                    )

            # Create edges only where both parent and child are in subtree
            for parent, children in parent_map.items():
                if parent not in node_ids:
                    continue
                for child in children:
                    if child not in node_ids:
                        continue
                    dot.edge(node_ids[parent], node_ids[child])

            filename = os.path.join(directory, f"query_{qi}_subtree.png")
            filepath = os.path.join(directory, f"query_{qi}_subtree")
            dot.render(filepath, cleanup=True)
            print(f"Saved: {filename}")
