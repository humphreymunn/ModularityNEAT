import q_score_measurement

''' NEAT network graph representation for plotting and calculating Q-Score. '''
class NeatGraph():

    '''
    genome: neat genome object.
    input_nodes: # of input nodes (these are not included in genome.nodes).
    '''
    def __init__(self, genome, input_nodes = 24):
        self.genome = genome
        self.input_nodes = input_nodes
        self.node_count = len(genome.nodes) + input_nodes
        self.edge_count = len(genome.connections)
        self.edges = self.init_edges(genome.connections)
        self.q_score, self.groups = self.calc_q_score()

    def __iter__(self):
        self.current_edge_idx = 0
        return self

    ''' Iterator to iterator over edges. '''
    def __next__(self):
        self.current_edge_idx += 1
        if self.current_edge_idx <= self.edge_count:
            return self.edges[self.current_edge_idx - 1]
        else:
            raise StopIteration

    class NeatEdge():
        def __init__(self, source, target):
            self.source = source
            self.target = target

        def __str__(self):
            return "(" + str(self.source) + "," + str(self.target) + ")"

    '''
    connections: dictionary of neat connections.
    Returns NeatEdges where source and target are scaled to be in [0, self.node_count].
    '''
    def init_edges(self, connections):
        edges = []
        sorted_ids = sorted(list(range(-self.input_nodes,0)) + [node.key for node in self.genome.nodes.values()])
        for conn in connections.values():
            if not conn.enabled:
                self.edge_count -= 1
                continue
            source, target = conn.key
            source = sorted_ids.index(source)
            target = sorted_ids.index(target)
            edges.append(self.NeatEdge(source, target))
        return edges

    def calc_q_score(self):
        Q, groups = q_score_measurement.Q(self)
        return Q, groups

    def plot_graph(self):
        pass

