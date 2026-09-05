def get_neighbors(graph, node):
    return set(graph.neighbors(node))

def shared_neighbors(graph, node_a, node_b):
    neighbors_a = get_neighbors(graph, node_a)
    neighbors_b = get_neighbors(graph, node_b)

    return len(neighbors_a & neighbors_b)


def jaccard_similarity(graph, node_a, node_b):
    neighbors_a = get_neighbors(graph, node_a)
    neighbors_b = get_neighbors(graph, node_b)

    union = neighbors_a | neighbors_b

    if len(union) == 0:
        return 0.0

    intersection = neighbors_a & neighbors_b

    return len(intersection) / len(union)

