import itertools
import json
import operator
import pathlib
import pickle
from typing import Callable

import networkx as nx
import numpy as np
from nltk.tree import Tree
from tqdm import tqdm

PATH_ROOT = pathlib.Path(__file__).parent.parent
PATH_DATASETS = PATH_ROOT / "datasets"


# Convert NLTK tree to Networkx Graph and then to LCA ##################################


def add_edges(G: nx.DiGraph, node: Tree) -> None:
    """
    Add edges to a directed graph recursively based on a tree structure.

    Args:
        G (nx.DiGraph): A directed graph.
        node (Tree): The root node of the tree.

    Returns:
        None
    """
    for child in node:
        if isinstance(child, str):
            G.add_edge(node.label(), child)
        else:
            G.add_edge(node.label(), child.label())
            add_edges(G, child)


def max_path_length(G: nx.DiGraph, node: str) -> int:
    """
    Calculate the maximum path length from a given node to a leaf in
    a directed graph.

    Args:
        G (nx.DiGraph): The directed graph.
        node (str): The starting node.

    Returns:
        int: The maximum path length.
    """
    # Base case: if the node is a leaf, return 0
    if G.out_degree(node) == 0:
        return 0

    # Initialize the maximum path length
    max_length = 0

    # Iterate over the children of the current node
    for child in G.successors(node):
        # Calculate the path length from the child to a leaf
        path_length = max_path_length(G, child) + 1

        # Update the maximum path length if necessary
        if path_length > max_length:
            max_length = path_length

    return max_length


def G_to_lca(G: nx.DiGraph, str_to_int: Callable) -> np.ndarray:
    """
    Convert a directed graph to a Least Common Ancestor (LCA) matrix.

    The LCA matrix is a square matrix where each element (i, j)
    represents the level of the least common ancestor for the ith and
    jth leaves in the graph.

    Args:
        G (nx.DiGraph): The directed graph.
        str_to_int (Callable): A function to convert node labels from
        strings to integers.

    Returns:
        np.ndarray: The LCA matrix.
    """
    # Calculate the heights of all nodes in the graph
    node_heights = {}
    for node in tqdm(G.nodes()):
        height = max_path_length(G, node)
        node_heights[node] = height

    # Get the leaves in the graph
    leaves = [node for node in G.nodes() if G.degree(node) == 1]
    num_leaves = len(leaves)

    # Generate all pairs of leaves
    leaves_x_leaves = itertools.product(leaves, leaves)

    # Calculate the lowest common ancestor for each pair of leaves
    lcas = nx.all_pairs_lowest_common_ancestor(G, leaves_x_leaves)

    # Initialize the LCA matrix
    lca_matrix = np.zeros((num_leaves, num_leaves), dtype=int)

    # Fill in the LCA matrix with the levels of the lowest common ancestors
    for (u, v), lca in tqdm(lcas, total=num_leaves * num_leaves):
        u, v = str_to_int(u), str_to_int(v)
        lca_matrix[u, v] = node_heights[lca]

    return lca_matrix


# Convert LCA to hierarchy and viceversa ###############################################


def lca_to_hierarchy(lca: np.ndarray) -> np.ndarray:
    """
    Converts a Least Common Ancestor (LCA) matrix to a hierarchy matrix.

    The hierarchy matrix is a matrix where each row represents the ancestor hierarchy
    of a class.

    Args:
        lca (np.array or torch.Tensor): A square matrix where each element (i, j)
            represents the level of the least common ancestor for classes i and j.

    Returns:
        A numpy array containing the hierarchy matrix.
    """
    # Make a copy to avoid inplace operations
    lca = np.array(lca, dtype=int)

    # Number of hierarchy levels (L)
    # Number of finer classes (C)
    L, C = lca.max(), len(lca)

    hierarchy = -np.ones((L, C), dtype=int)

    for level in range(L):
        # Find all siblings at `level`,
        # reverse to be consistence at level 0
        siblings = np.unique(lca == level, axis=0)[::-1]

        # Generate labeler
        labeler = np.arange(len(siblings), dtype=int)

        # Apply labels to siblings with labeler
        labels = labeler @ siblings

        # Add labels to hierarchy
        hierarchy[level] = labels

        # Update lca for next iteration
        lca[lca == level] += 1

    return hierarchy


def hierarchy_to_lca(hierarchy: np.ndarray) -> np.ndarray:
    """
    Converts a hierarchy to a Least Common Ancestor (LCA) matrix.

    The LCA matrix is a square matrix where each element (i, j) represents
    the level of the least common ancestor for classes i and j.

    Args:
        hierarchy (np.array or torch.Tensor): A matrix where each row represents
            the ancestor hierarchy of a class.

    Returns:
        A square numpy array containing the LCA matrix.
    """
    # Number of hierarchy levels (L)
    # Number of finer classes (C)
    L, C = hierarchy.shape

    lca = np.full((C, C), L, dtype=int)

    for level in hierarchy:
        for row, coarse in zip(lca, level):
            for index, value in enumerate(level):
                if coarse == value:
                    row[index] -= 1
    return lca


# CIFAR100 #############################################################################

path_hierarchy = PATH_DATASETS / "CIFAR100" / "hierarchy"

with open(path_hierarchy / "tree.pkl", "rb") as f:
    nltk_tree = pickle.load(f)

G = nx.DiGraph()
add_edges(G, nltk_tree)

lca = G_to_lca(G, lambda x: int(x.split("-")[-1]))
np.save(path_hierarchy / "lca.npy", lca)

hierarchy = lca_to_hierarchy(lca)
np.save(path_hierarchy / "hierarchy.npy", hierarchy)

# iNaturalist19 ########################################################################

path_hierarchy = PATH_DATASETS / "iNaturalist19" / "hierarchy"
path_classes = PATH_DATASETS / "iNaturalist19" / "classes"

with open(path_hierarchy / "tree.pkl", "rb") as f:
    nltk_tree = pickle.load(f)

G = nx.DiGraph()
add_edges(G, nltk_tree)

with open(path_classes / "dir_to_int.json") as f:
    dir_to_int = json.load(f)

lca = G_to_lca(G, lambda x: operator.getitem(dir_to_int, x))
np.save(path_hierarchy / "lca.npy", lca)

hierarchy = lca_to_hierarchy(lca)
np.save(path_hierarchy / "hierarchy.npy", hierarchy)

# tieredImageNet  ######################################################################

path_hierarchy = PATH_DATASETS / "tieredImageNet" / "hierarchy"
path_classes = PATH_DATASETS / "tieredImageNet" / "classes"

with open(path_hierarchy / "tree.pkl", "rb") as f:
    nltk_tree = pickle.load(f)

G = nx.DiGraph()
add_edges(G, nltk_tree)

with open(path_classes / "dir_to_int.json") as f:
    dir_to_int = json.load(f)

lca = G_to_lca(G, lambda x: operator.getitem(dir_to_int, x))
np.save(path_hierarchy / "lca.npy", lca)

hierarchy = lca_to_hierarchy(lca)
np.save(path_hierarchy / "hierarchy.npy", hierarchy)
