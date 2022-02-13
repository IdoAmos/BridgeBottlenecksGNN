import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric.utils as utils
from torch_geometric.utils import to_networkx

class Connect:
    def __init__(self, graph):
        self.data = graph.clone()
        self.G = to_networkx(graph, to_undirected=True)

    def reboot(self):
        """
        Reset self.G to source graph
        :return:
        """
        self.G = to_networkx(self.data, to_undirected=True)

    def to_torch_geo(self):
        """
        Move modified graph back to torch geometric graph
        :return: torch geometric data
        """
        graph = utils.from_networkx(self.G)
        graph.x = self.data.x
        graph.y = self.data.y
        return graph

    def reduce_graph(self):
        """
        Delete bridges from self.G
        :return:
        """
        bridges = nx.bridges(self.G)
        bridge_lst = []
        for edge in bridges:
            self.G.remove_edge(edge[0], edge[1])
            bridge_lst.append(edge)
        return bridge_lst

    def complete_cc(self):
        """
        Make each CC of self.graph as complete graph
        :return:
        """
        cc = [self.G.subgraph(c).copy() for c in nx.connected_components(self.G)]

        cliques = []
        for g in cc:
            cliques.append(nx.complete_graph(g))
        return cliques

    def union_of_cc_completions(self):
        """
        Generate the (disconnected) graph from the union of the completed CC of self.graph
        :return:
        """
        cliques = self.complete_cc()
        self.G = nx.compose_all(cliques)

    def complement(self):
        """
        set self.G as the complement of self.G
        :return:
        """
        self.G = nx.complement(self.G)

    def reduced_cliques_complement(self):
        bridges = self.reduce_graph()

        if len(bridges) == 0:
            self.G = nx.complete_graph(self.G) # if no bridges use original FA matrix TODO: DECIDE IF SHOULD USE FULL ADJ. OR SOURCE ADJ.
        else:
            self.union_of_cc_completions()
            self.complement()

        graph = self.to_torch_geo()
        return graph

    def visualize_graph(self, labels=True, title=None):
        """
        Plot self.G
        :param labels:
        :return:
        """
        plt.figure(figsize=(10, 10))
        plt.xticks([])
        plt.yticks([])
        nx.draw_networkx(self.G, pos=nx.spring_layout(self.G, seed=42), with_labels=labels,
                         node_color=torch.zeros(size=(len(self.G.nodes), 1)), cmap="Set2")
        plt.suptitle(title) if title is not None else ''
        plt.show()

    def visualize_source(self, labels=True):
        """
        Plot original graph
        :param labels:
        :return:
        """
        G = to_networkx(self.data, to_undirected=True)
        plt.figure(figsize=(10, 10))
        plt.xticks([])
        plt.yticks([])
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=labels,
                         node_color=torch.zeros(size=(len(G.nodes), 1)), cmap="Set2")
        plt.show()

    def visualize_process(self, data):
        plt.close('all')
        self.data = data.clone()
        self.G = to_networkx(data, to_undirected=True)
        self.visualize_graph(title='Source')
        bridges = self.reduce_graph()
        self.visualize_graph(title='reduced graph for bridges:{}'.format(bridges))
        print("Found {} bridges in graph".format(len(bridges)))
        if len(bridges) == 0:
            self.G = nx.complete_graph(self.G) # if no bridges use original FA matrix TODO: DECIDE IF SHOULD USE FULL ADJ. OR SOURCE ADJ.
        else:
            self.union_of_cc_completions()
            self.visualize_graph(title='Union of cliques')
            self.complement()
        self.visualize_graph(title='final')
        self.reboot()