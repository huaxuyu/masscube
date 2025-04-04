# Author: Huaxu Yu

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
from .annotation import feature_to_feature_search


def network_analysis(feature_list, annotation_type="hybrid_and_identity", feature_quality="all", show_node_name=False, output=False):
    """
    A function to plot a network graph.

    Parameters
    ----------
    feature_list : list of Feature objects
        A list of features to be plotted.
    annotation_type : str
        Type of annotation to be plotted. Default is "all".
        "all" - all the features with MS2 spectra.
        "hybrid_and_identity" - features with identity and hybrid annotation.
        "identity_only" - only features with identity annotation.
        "hybrid_only" - only features with hybrid annotation.
    feature_quality : str
        Quality of features to be plotted. Default is "all".
        "all" - all the features.
        "good" - only good features (quality=="good").
        "bad" - only bad features (quality=="bad peak shape").
    """

    # prepare feature list
    selected_features = _prepare_feature_list_for_network(feature_list, annotation_type, feature_quality)

    df = feature_to_feature_search(selected_features)

    hybrid_features = [f for f in selected_features if f.annotation_mode == "hybrid_search"]

    identity_search_names = [f.annotation for f in selected_features if f.annotation_mode == "identity_search"]

    if len(hybrid_features) > 0:
        for f in hybrid_features:
            if f.annotation in identity_search_names:
                df.loc[len(df)] = [f.network_name, f.annotation, f.similarity, f.id, "DB"]
            else:
                df.loc[len(df)] = [f.network_name, "DB_"+f.annotation, f.similarity, f.id, "DB"]

    # Create a new graph
    G = nx.Graph()

    # prepare nodes
    nodes = df["feature_name_1"].tolist() + df["feature_name_2"].tolist()
    nodes = list(set(nodes))

    # prepare edges
    edges = []
    for i in range(len(df)):
        edges.append((df["feature_name_1"][i], df["feature_name_2"][i]))

    # define node colors: identity - green, hybrid - "#8BABD3", database - gray, unknown - pink
    node_color = []
    for n in nodes:
        if n.startswith("hybrid"):
            node_color.append("#FEFAE0")
        elif n.startswith("unknown"):
            node_color.append("pink")
        elif n.startswith("DB"):
            node_color.append("#283618")
        else:
            node_color.append("#BC6C25")

    # define edge colors as a gradient of similarity
    edge_color = _edge_color_gradient(df["similarity"])

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # make plot
    pos = nx.spring_layout(G, iterations=25)  # positions for all nodes
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=2)
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=40, alpha=0.5, edgecolors="black", linewidths=0.5)
    if show_node_name:
        nx.draw_networkx_labels(G, pos, font_size=8, font_family="arial", labels={n: n.split("_")[-1] for n in nodes if n.startswith("identity")})

    # hide outer frame
    plt.box(False)
    plt.rcParams['font.size'] = 12
    # add legend with arial font
    plt.legend(handles=[plt.Line2D([0], [0], color="#BC6C25", marker="o", lw=0, markersize=7, label="Identity", markeredgewidth=0.5, markeredgecolor="black", alpha=0.5),
                        plt.Line2D([0], [0], color="#FEFAE0", marker="o", lw=0, markersize=7, label="Hybrid", markeredgewidth=0.5, markeredgecolor="black", alpha=0.5),
                        plt.Line2D([0], [0], color="#283618", marker="o", lw=0, markersize=7, label="Database", markeredgewidth=0.5, markeredgecolor="black", alpha=0.5)],
               loc="upper left", bbox_to_anchor=(0.9, 1))

    if output:
        plt.savefig(output, dpi=1000, bbox_inches="tight")
        plt.close()
        df.to_csv(output.replace(".png", ".csv"), index=False)
    else:
        plt.show()


def _prepare_feature_list_for_network(feature_list, annotation_type="hybrid_and_identity", feature_quality="all"):
    """
    A function to prepare the feature list for plotting.
    
    Parameters
    ----------
    feature_list : list of Feature objects
        A list of features to be plotted.
    annotation_type : str
        Type of annotation to be plotted. Default is "all".
        "all" - all the features with MS2 spectra.
        "hybrid_and_identity" - features with identity and hybrid annotation.
        "identity_only" - only features with identity annotation.
        "hybrid_only" - only features with hybrid annotation.
    feature_quality : str
        Quality of features to be plotted. Default is "all".
        "all" - all the features.
        "good" - only good features (quality=="good").
        "bad" - only bad features (quality=="bad peak shape").

    Returns
    -------
    selected_features : list of Feature objects
        A list of features to be plotted.
    """
    
    selected_features = [f for f in feature_list if f.best_ms2 is not None]

    if annotation_type == "all":
        selected_features = feature_list
    elif annotation_type == "hybrid_and_identity":
        selected_features = [f for f in feature_list if f.annotation_mode in ["identity_search", "hybrid_search"]]
    elif annotation_type == "identity_only":
        selected_features = [f for f in feature_list if f.annotation_mode == "identity_search"]
    elif annotation_type == "hybrid_only":
        selected_features = [f for f in feature_list if f.annotation_mode == "hybrid_search"]
    else:
        raise ValueError("Invalid annotation_type: {}".format(annotation_type))
    
    if feature_quality == "all":
        pass
    elif feature_quality == "good":
        selected_features = [f for f in selected_features if f.quality == "good"]
    elif feature_quality == "bad":
        selected_features = [f for f in selected_features if f.quality == "bad peak shape"]
    else:
        raise ValueError("Invalid feature_quality: {}".format(feature_quality))
    
    for f in selected_features:
        if f.annotation_mode == "identity_search":
            f.network_name = "identity_{}".format(f.id) + "_" + f.annotation
        elif f.annotation_mode == "hybrid_search":
            f.network_name = "hybrid_{}".format(f.id)
        else:
            f.network_name = "unknown_{}".format(f.id)
    
    return selected_features


def _edge_color_gradient(similarity_array, color_1="lightgrey", color_2="black"):
    """
    A function to generate a list of edge colors based on the similarity scores.
    """

    colors = []

    similarity_array = np.array(similarity_array)

    similarity_array = (np.max(similarity_array) - similarity_array) / (np.max(similarity_array) - np.min(similarity_array))

    for s in similarity_array:
        colors.append(_color_gradient(s, color_1, color_2))

    return colors


def _color_gradient(s, color_1, color_2):
    """
    A function to generate a color based on the similarity score.
    """

    color_1 = mcolors.to_rgb(color_1)
    color_2 = mcolors.to_rgb(color_2)

    r = color_1[0] + s * (color_2[0] - color_1[0])
    g = color_1[1] + s * (color_2[1] - color_1[1])
    b = color_1[2] + s * (color_2[2] - color_1[2])

    return (r, g, b)