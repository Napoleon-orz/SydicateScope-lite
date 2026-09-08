import networkx as nx


def normalize(values):
    """
    Normalize dictionary values to 0-1.
    """

    if not values:
        return {}

    maximum = max(values.values())

    if maximum == 0:
        return {
            node: 0.0
            for node in values
        }

    return {
        node: value / maximum
        for node, value in values.items()
    }


def calculate_fir_evidence(graph):
    """
    Measures how strongly each person is supported
    by FIR co-mention evidence.
    """

    scores = {}

    for node in graph.nodes():

        fir_weight = 0.0

        for _, _, data in graph.edges(
            node,
            data=True
        ):

            fir_weight += data.get(
                "fir_weight",
                0.0
            )

        scores[node] = fir_weight

    return normalize(scores)


def calculate_multi_source_evidence(graph):
    """
    Rewards people who have multiple types of
    investigative evidence.

    Sources considered:
    - FIR
    - CDR
    - Transaction
    """

    scores = {}

    for node in graph.nodes():

        sources = set()

        for _, _, data in graph.edges(
            node,
            data=True
        ):

            if data.get("fir_weight", 0) > 0:
                sources.add("FIR")

            if data.get("cdr_weight", 0) > 0:
                sources.add("CDR")

            if data.get("txn_weight", 0) > 0:
                sources.add("TRANSACTION")

        scores[node] = len(sources) / 3.0

    return scores


def calculate_influence_scores(graph):

    # --------------------------------------------------
    # CENTRALITY
    # --------------------------------------------------

    degree = normalize(
        nx.degree_centrality(graph)
    )

    betweenness = normalize(
        nx.betweenness_centrality(graph)
    )

    pagerank = normalize(
        nx.pagerank(graph)
    )

    # --------------------------------------------------
    # INVESTIGATION EVIDENCE
    # --------------------------------------------------

    fir_evidence = calculate_fir_evidence(
        graph
    )

    multi_source = calculate_multi_source_evidence(
        graph
    )

    # --------------------------------------------------
    # FINAL SCORE
    #
    # The old version summed raw topology (degree +
    # betweenness + pagerank, 70% of the score) with
    # evidence (fir_evidence + multi_source, 30%). That let
    # a purely-noisy, single-channel hub (lots of CDR/txn
    # chatter, zero FIR evidence) outscore a person who is
    # genuinely corroborated across CDR + transaction + FIR
    # channels but has a smaller raw footprint in any one of
    # them - exactly backwards from what this project is
    # supposed to demonstrate (README2.md's "central only
    # when combined" signal).
    #
    # Fix: gate raw topology by evidence diversity before
    # weighting it in. A node touching only one channel keeps
    # 40% of its topology score; a node corroborated across
    # all three channels keeps the full 100%. FIR evidence
    # (the hardest evidence to fabricate via random noise) is
    # weighted independently, more heavily than before.
    # --------------------------------------------------

    scores = {}

    for node in graph.nodes():

        topology = (
            0.45 * degree.get(node, 0)
            + 0.35 * betweenness.get(node, 0)
            + 0.20 * pagerank.get(node, 0)
        )

        multi_source_score = multi_source.get(node, 0)

        gated_topology = topology * (0.4 + 0.6 * multi_source_score)

        scores[node] = (

            0.55 * gated_topology

            + 0.35 * fir_evidence.get(node, 0)

            + 0.10 * multi_source_score

        )

    return scores


def rank_influential_nodes(
    graph,
    top_n=10,
    person_only=False
):
    """
    person_only restricts the ranking to PERSON nodes.

    This matters once VEHICLE/ORGANIZATION/LOCATION nodes carry real
    edges (see graph.add_vehicle_edges / add_organization_edges /
    add_location_edges): a single busy cell tower can rack up a
    higher raw degree centrality than any one person, since dozens of
    different people ping it. Left unfiltered, that tower would edge
    out real suspects from a "Key Individuals" panel - which the PRD
    (FR-22) defines as people, not places. Sizing nodes on the graph
    itself is a separate concern and should keep using the unfiltered
    ranking, since a structurally important location/org is still
    worth drawing bigger there.
    """

    scores = calculate_influence_scores(
        graph
    )

    if person_only:
        scores = {
            node: score
            for node, score in scores.items()
            if graph.nodes[node].get("entity_type", "").upper() == "PERSON"
        }

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_n]