"""
Wrapper to make CobwebClusterer class ready for BERTopic Library!!

TODO not done yet, hierarchical could be valuable, but also not sure how to do it

https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html
"""

import pandas as pd

class BERTopicHierarchicalWrapper:
    def __init__(self, docs, bertopic_model, linkage_function = None, cobweb_clusterer = None):
        if cobweb_clusterer is not None:
            self._from_cobweb_clusterer(docs, cobweb_clusterer, bertopic_model)
        else:
            self._from_bertopic_hierarchical(docs, bertopic_model, linkage_function)
        
    def _from_bertopic_hierarchical(self, docs, bertopic_model, linkage_function):
        ## Big TODO
        ## Need to check if topics returns the topics like we want or if we need to recode everything
        hierarchical_topics = bertopic_model.hierarchical_topics(docs, linkage_function=linkage_function)
        parents = set(hierarchical_topics["Parent_ID"].tolist())
        left_children = set(hierarchical_topics["Child_Left_ID"].tolist())
        right_children = set(hierarchical_topics["Child_Right_ID"].tolist())
        all_children = left_children.union(right_children)
        leaves = all_children.difference(parents)

        adjacency = {}
        topics_map = {}
        for _, row in hierarchical_topics.iterrows():
            parent_id = row["Parent_ID"]
            adjacency[parent_id] = [row["Child_Left_ID"], row["Child_Right_ID"]]
            topics_map[parent_id] = row["Topics"]

        levels = {}
        for leaf in leaves:
            levels[leaf] = 0

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

        rows = []
        for leaf in sorted(leaves):
            rows.append({
                "Node_ID": leaf,
                "Topics": [leaf],
                "Level": levels.get(leaf, 0),
                "children_ids": []
            })
        for parent in sorted(adjacency.keys()):
            rows.append({
                "Node_ID": parent,
                "Topics": topics_map.get(parent),
                "Level": levels.get(parent),
                "children_ids": adjacency.get(parent, [])
            })

        df = pd.DataFrame(rows).sort_values(["Level", "Node_ID"]).reset_index(drop=True)
        self.hierachical_topics = df


    ## from CobwebClusterer
    def _from_cobweb_clusterer(self, docs, cobweb_clusterer, bertopic_model):
        ## This Copilot can handle by mixing bertopic code and the code from my topicmodeling class
        doc_map, children_map = cobweb_clusterer._create_node_doc_assignment()
        