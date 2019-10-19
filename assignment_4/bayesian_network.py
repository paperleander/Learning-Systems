# Bayesian Network
# Assignment 4
# Moist Grass
#                -........-                             `-..............-`
#                / -.````..                             /..``````.`.```  /
#                / Rain:`..                             /.: Sprinkler    /
#                :..:..:.:`                             .-.-:...........-.
#                   `:`  :.                                  .:
#                  .-      .:                               :.
#                 `:`        :.                           .:
#                .-           .-                         :.
#               :`              :.                     .:
#              .-                .-                   :.
#            :.                    :`               .:
#           -:                       ./            /.
# :-........-.......:            :......-...-.....--
# /: /.`. - -`.....`/            ..::..-.....`.    /
# / ``Watson's `.   /            ....Holmes'-..    /
# / -.`..``` --.``. /            ..`....```` -:.`../
# / -.Grass Wet..-  /            .-`-Grass Wet-`-../
# .-................-            -................-.

from collections import defaultdict

nodes = ["Watson", "Holmes", "Rain", "Sprinkler"]


class BayesianNetwork:
    def __init__(self):
        self.graph = defaultdict(Node)

        # Make nodes
        for name in nodes:
            self.graph[name] = Node(name)

        # Connect nodes
        self.connect("Rain", "Watson")
        self.connect("Rain", "Holmes")
        self.connect("Sprinkler", "Holmes")

        # Add probabilities
        self.graph["Rain"].table["t"] = 0.2
        self.graph["Rain"].table["f"] = 0.8
        self.graph["Rain"].p = 0.2

        self.graph["Sprinkler"].table["t"] = 0.3
        self.graph["Sprinkler"].table["f"] = 0.7
        self.graph["Sprinkler"].p = 0.3

        self.graph["Watson"].table["t"] = 0.2
        self.graph["Watson"].table["f"] = 0.8
        self.graph["Watson"].p = 0.2

        self.graph["Holmes"].table["t"] = 0.25
        self.graph["Holmes"].table["f"] = 0.75
        self.graph["Holmes"].p = 0.25

    def connect(self, parent, child):
        # Minimize emotional problems
        self.graph[parent].children[child] = self.graph[child]
        self.graph[child].parents[parent] = self.graph[parent]

    def get_graph(self):
        return self.graph

    def P(self, a):
        return self.graph[a].p

    def P(self, a, b):
        return

    def estimate(self, x, wx):
        """
        - Let Wx = the states of all other variables except x.
        - Let the Markov Blanket of a node be all of its parents,
        children and parents of children.
        - Distribution of each node, x, conditioned upon Wx can be computed
        locally from their own probability with their children’s :
        P(a|Wa) = alpha . P(a) . P(b|a) . P(c|a)
        P(b|Wb) = alpha . P(b|a) . P(d|b,c)
        P(c|Wc) = alpha . P(c|a) . P(d|b,c) . P(e|c)
        - alpha makes the probabilities add up to 1
        :param a: to estimate
        :param b: known parameter
        :return: probability of x
        """
        p = 0
        x = self.graph[x]
        a = self.graph[wx[0]]
        b = self.graph[wx[1]]
        c = self.graph[wx[2]]

        return p


class Node:
    def __init__(self, name):
        self.name = name
        self.parents = defaultdict(Node)
        self.children = defaultdict(Node)
        self.table = defaultdict(float)
        self.p = None


if __name__ == '__main__':
    bn = BayesianNetwork()

    for x in nodes:
        # Get all other nodes than x
        wx = nodes.copy()
        wx.remove(x)

        # Estimate P(x | Wx)
        bn.estimate(x, wx)

    print(bn.get_graph())

