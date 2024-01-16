import numpy as np

network = []

network_colors = []

def rand():
    return 2 * np.random.random() - 1

input = rand()

def update_colors():
    network_colors.append([network[i].get_last_value() for i in range(len(network))])
    print(network_colors[-1])

class Neuron:
    def __init__(self, index, other_neuron_idx1, other_neuron_idx2, weight, bias, is_input=False):
        self.index = index
        self.weight = weight # (weight + 1)/2
        self.bias = bias
        self.other_neuron_idx1 = other_neuron_idx1
        self.other_neuron_idx2 = other_neuron_idx2
        self.is_input = is_input
        self.last_value = bias
        if is_input:
            self.last_value = input

    def feedforward(self, recursionLevel=0):
        if self.is_input:
            return input
        if recursionLevel > len(network):
            print("Recursion limit reached!!!")
            return self.last_value
        value = 0
        if self.other_neuron_idx1 != None:
            value += self.weight * network[self.other_neuron_idx1].feedforward(recursionLevel+1)
        if self.other_neuron_idx2 != None:
            value += self.weight * network[self.other_neuron_idx2].feedforward(recursionLevel+1)
            # value /= 2
        value += self.bias
        value = activation(value)
        self.last_value = value
        print(self.last_value, self.other_neuron_idx1, self.other_neuron_idx2)
        update_colors()
        return value
        # return sigmoid(value)
    
    def get_last_value(self):
        return self.last_value
    def get_index(self):
        return self.index
    
    def __str__(self):
        return f"Neuron {self.index} with weight {self.weight} and bias {self.bias} and last value {self.last_value}"
    
def activation(x):
    return 1 / (1 + np.exp(-x))

# create first two neurons
network.append(Neuron(0, None, None, rand(), rand(), is_input=True))
network.append(Neuron(1, 0, None, rand(), rand()))

# add neurons to network
for i in range(2, 20):
    # if i % 5 == 0:
    #     network.append(Neuron(i, i-1, int(i/5), rand(), rand()))
    network.append(Neuron(i, i-1, None, rand(), rand()))

# make the graph a bit more interesting by adding some cycles
network[0].other_neuron_idx1 = len(network)-1
# network[0].other_neuron_idx2 = 4
network[10].other_neuron_idx2 = 15

network[0].other_neuron_idx2 = 10

# network[0].other_neuron_idx2 = 8


update_colors()
print(*network)
print(network[-1].feedforward())
print(*network)

import matplotlib.pyplot as plt; plt.close('all')
import networkx as nx
from matplotlib.animation import FuncAnimation

def animate_nodes(G, node_colors, pos=None, *args, **kwargs):
    # define graph layout if None given
    if pos is None:
        # pos = nx.spring_layout(G)
        # pos = nx.circular_layout(G)
        # pos = nx.shell_layout(G)
        pos = nx.spectral_layout(G)

    # draw graph
    nodes = nx.draw_networkx_nodes(G, pos, *args, **kwargs)
    edges = nx.draw_networkx_edges(G, pos, *args, **kwargs)
    plt.axis('off')

    def update(ii):
        # nodes are just markers returned by plt.scatter;
        # node color can hence be changed in the same way like marker colors
        nodes.set_array(node_colors[ii])
        return nodes,

    fig = plt.gcf()
    animation = FuncAnimation(fig, update, interval=50, frames=len(node_colors), blit=True)
    return animation

# make nx graph

graph = nx.DiGraph()

for i in range(0, len(network)):
    graph.add_node(i)
    if network[i].other_neuron_idx1 != None:
        graph.add_edge(network[i].other_neuron_idx1, i)
    if network[i].other_neuron_idx2 != None:
        graph.add_edge(network[i].other_neuron_idx2, i)

animation = animate_nodes(graph, network_colors)
animation.save('test.gif', writer='imagemagick', savefig_kwargs={'facecolor':'white'}, fps=10)



