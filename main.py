import numpy as np

network = []

# network_colors = []

node_colors = []

def rand():
    return 2 * np.random.random() - 1

input = rand()

def update_colors():
    global node_colors
    node_colors = [network[i].value for i in range(len(network))]
    # print(network_colors[-1])

def step():
    # for i in range(len(network)):
    #     network[i].value = network[i].bias
    for i in range(len(network)):
        for n in network[i].other_neuron_idxs:
            print(i, n, network[i].weight, network[i].value, network[n].value)
            network[n].value += network[i].weight * network[i].value
    for i in range(len(network)):
        network[i].value = activation(network[i].value)
    update_colors()
            

class Neuron:
    def __init__(self, index, other_neuron_idxs, weight, bias):
        self.index = index
        self.weight = weight # (weight + 1)/2
        self.bias = bias
        self.other_neuron_idxs = other_neuron_idxs
        # self.is_input = is_input
        self.value = activation(bias)
        # if is_input:
        #     self.last_value = input

    # def feedforward(self, input, recursionLevel=0):
    #     print("State start:", network_colors[-1])
    #     # if self.is_input:
    #     #     return input
    #     # if recursionLevel > len(network):
    #     #     print("Recursion limit reached!!!")
    #     #     return self.last_value
    #     if recursionLevel > 1:
    #         return
    #     value = activation(self.bias + self.weight * input)
    #     self.last_value = value
    #     # update_colors()
    #     print("State end:  ", network_colors[-1])

    #     # print("Updated colors", recursionLevel)
    #     if self.other_neuron_idx1 != None:
    #         network[self.other_neuron_idx1].feedforward(value, recursionLevel+1)
    #         # value += self.weight * network[self.other_neuron_idx1].feedforward(recursionLevel+1)
    #     if self.other_neuron_idx2 != None:
    #         network[self.other_neuron_idx2].feedforward(value, recursionLevel+1)
    #         # value += self.weight * network[self.other_neuron_idx2].feedforward(recursionLevel+1)
    #     # update_colors()
    #     # self.continue_forward(value)
    #         # value /= 2
    #     # print(self.last_value, self.other_neuron_idx1, self.other_neuron_idx2)
    #     return value
    
    # def continue_forward(self, value, recursionLevel=0):
    #     update_colors()
    #     if recursionLevel > 0:
    #         return
    #     if self.other_neuron_idx1 != None:
    #         network[self.other_neuron_idx1].continue_forward(value, recursionLevel+1)
    #     if self.other_neuron_idx2 != None:
    #         network[self.other_neuron_idx2].continue_forward(value, recursionLevel+1)
        # if self.other_neuron_idx1 != None:
        #     network[self.other_neuron_idx1].feedforward(value, recursionLevel+1)
        #     # value += self.weight * network[self.other_neuron_idx1].feedforward(recursionLevel+1)
        # if self.other_neuron_idx2 != None:
        #     network[self.other_neuron_idx2].feedforward(value, recursionLevel+1)
    
    # def get_last_value(self):
    #     return self.last_value
    # def get_index(self):
    #     return self.index
    
    def __str__(self):
        return f"Neuron {self.index} with weight {self.weight} and bias {self.bias} and value {self.value}"
    
def activation(x):
    # return np.exp(-x*x)
    # return np.log(1 + np.exp(x))
    return np.tanh(x)
    # return 1/(1 + np.exp(-x))

# create first two neurons
# network.append(Neuron(0, [1], rand(), rand()))

# network.append(Neuron(1, [4], rand(), rand()))

# network.append(Neuron(2, [1], rand(), rand()))

# network.append(Neuron(3, [1], rand(), rand()))

# network.append(Neuron(4, [0], rand(), rand()))

for i in range(0, 10):
    network.append(Neuron(i, [i+1], rand(), rand()))

network.append(Neuron(len(network), [0], rand(), rand()))

# for i in range(3, 7):
#     network.append(Neuron(i, [0], rand(), rand()))

# network.append(Neuron(3, [], rand(), rand()))

# for i in range(3, 4):
#     # if i % 3 == 0:
#     #     network.append(Neuron(i, i-1, i-2, rand(), rand()))
#     network.append(Neuron(i, i+1, None, rand(), rand()))

# # # network.append(Neuron(len(network)-2, len(network)-1, None, rand(), rand()))
# network.append(Neuron(len(network), 0, None, rand(), rand()))

# # make a circle
# for i in range(0, 5):
#     network.append(Neuron(i, None, 0, rand(), rand()))
# network.append(Neuron(len(network), 0, None, rand(), rand()))

# connect the 0th neuron to the middle
# network[0].other_neuron_idx2 = 10

# complicate further
# network[10].other_neuron_idx2 = 14
# network[7].other_neuron_idx2 = 9
# network[9].other_neuron_idx2 = 9



# network.append(Neuron(1, 0, None, rand(), rand()))

# # add neurons to network
# for i in range(2, 10):
#     # if i % 5 == 0:
#     #     network.append(Neuron(i, i-1, int(i/5), rand(), rand()))
#     network.append(Neuron(i, i-1, i-2, rand(), rand()))

# make the graph a bit more interesting by adding some cycles
# network[0].other_neuron_idx1 = len(network)-1
# network[0].other_neuron_idx2 = 4
# network[10].other_neuron_idx2 = 15

# network[0].other_neuron_idx2 = 10

# network[0].other_neuron_idx2 = 8

print(*network)
# update_colors()

# step()
# step()
# step()
# step()
# step()
# network[0].feedforward(input)
# network[0].continue_forward(input)
# print(*network)
# print(network[-1].value)




import matplotlib.pyplot as plt; plt.close('all')
import matplotlib as mpl
import networkx as nx
from matplotlib.animation import FuncAnimation

cmap = mpl.cm.viridis

def mapper(x):
    # for i in range(len(x)):
    #     x[i] = x[i] * 1.5
    return x

def animate_nodes(G, pos=None, *args, **kwargs):
    # define graph layout if None given
    if pos is None:
        # pos = nx.spring_layout(G)
        # pos = nx.circular_layout(G)
        # pos = nx.shell_layout(G)
        pos = nx.spectral_layout(G)

    # draw graph
    nodes = nx.draw_networkx_nodes(G, pos,  *args, **kwargs)
    edges = nx.draw_networkx_edges(G, pos, *args, **kwargs)
    plt.axis('off')

    # set sizes according to weight
    nodes.set_sizes([np.abs(network[i].weight)*64 for i in range(len(network))])
    nodes.set_cmap(cmap)

    def update(ii):
        step()
        # nodes are just markers returned by plt.scatter;
        # node color can hence be changed in the same way like marker colors
        nodes.set_array(mapper(node_colors))

        print("Updating colors", node_colors)

        return nodes,

    fig = plt.gcf()
    animation = FuncAnimation(fig, update, interval=50, frames=10000, blit=True)
    return animation

# make nx graph

graph = nx.DiGraph()

for i in range(0, len(network)):
    graph.add_node(i)
    for n in network[i].other_neuron_idxs:
        graph.add_edge(i, n)
    # if network[i].other_neuron_idx1 != None:
    #     graph.add_edge(i, network[i].other_neuron_idx1)
    # if network[i].other_neuron_idx2 != None:
    #     graph.add_edge(i, network[i].other_neuron_idx2)

animation = animate_nodes(graph)
plt.show()
# animation.save('./test.gif', writer='imagemagick', savefig_kwargs={'facecolor':'white'}, fps=2)



