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
        self.graph["Rain"].p = 0.2
        # self.graph["Rain"].prior["t"] = 0.2
        # self.graph["Rain"].prior["f"] = 0.8

        self.graph["Sprinkler"].p = 0.3
        # self.graph["Sprinkler"].prior["t"] = 0.3
        # self.graph["Sprinkler"].prior["f"] = 0.7

        # self.graph["Watson"].p = 0.2
        self.graph["Watson"].prior["t"] = 0.9
        self.graph["Watson"].prior["f"] = 0.05

        # self.graph["Holmes"].p = 0.25
        self.graph["Holmes"].prior["tt"] = 0.95
        self.graph["Holmes"].prior["tf"] = 0.9
        self.graph["Holmes"].prior["ft"] = 0.6
        self.graph["Holmes"].prior["ff"] = 0.05

    def connect(self, parent, child):
        # Minimize emotional problems
        self.graph[parent].children[child] = self.graph[child]
        self.graph[child].parents[parent] = self.graph[parent]

    def get_graph(self):
        return self.graph

    def p_a(self, a):
        return self.graph[a].p

    def p_a_given_b(self, a, b):
        return

    def p_a_given_b_c(self, a, b, c):
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
        :param x: to estimate
        :param wx: known parameter
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
        self.prior = defaultdict(float)
        self.p = None


if __name__ == '__main__':
    bn = BayesianNetwork()
    """
    for x in nodes:
        # Get all other nodes than x
        wx = nodes.copy()
        wx.remove(x)

        # Estimate P(x | Wx)
        bn.estimate(x, wx)
    """

    ##########################################
    # Hard-coding to learn Bayesian Networks #
    ##########################################

    # W - Watson
    # H - Holmes
    # R - Rain
    # S - Sprinkler

    # Probability of  Watson Crashes?
    # We don't know P(W), but we can calculate P(W|R)
    # P(W) = P(W|R)*P(R) + P(W|-R) * P(-R)
    p = 0.9 * 0.2 + 0.05 * 0.8
    print("Probability of Watson's Grass is wet:", p)  # 0.22

    # What if we want to calculate the probability of Rain given Watson's grass is wet?
    # Our arrows does not go in that direction, but we can use Bayes' Rule:
    # P(R|W) = P(W|R) * P(R) / P(W)
    p = 0.9 * 0.2 / 0.22
    print("Probability of Rain given Watson's Grass is wet:", p)  # 0.818
    # We started with probability of rain at 0.2
    # but after knowing that Watson's grass is wet, that probability was raised to 0.818

    # What about Holmes' Grass? (Given Watson's grass is wet)
    # Because of d-separation between Watson and Holmes given Rain, we go from this:
    # P(H|W) = P(H|W,R) * P(R|W) + P(H|W,-R) * P(-R|W), to this:
    # P(H|W) = P(H|R) * P(R|W) + P(H|-R) * P(-R|W)
    p = 0.9 * 0.818 + 0.1 * 0.182
    print("Probability of Holmes' grass is wet given Watson's Grass is wet:", p)  # 0.754

