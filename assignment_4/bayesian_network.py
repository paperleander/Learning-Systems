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

        # P(R)
        self.graph["Rain"].p = 0.2  # Chance of rain

        # P(S)
        self.graph["Sprinkler"].p = 0.1  # Chance of sprinkler

        # P(W|R)
        self.graph["Watson"].prior["t"] = 1.0  # If it rained, it is definitely wet
        self.graph["Watson"].prior["f"] = 0.2  # If it did not rain, it can still be wet by other reasons (cat piss?)

        # P(H|R,S)
        self.graph["Holmes"].prior["tt"] = 1.0  # Same goes for Holmes' grass even when sprinkler is also on
        self.graph["Holmes"].prior["tf"] = 1.0  # Still wet after raining
        self.graph["Holmes"].prior["ft"] = 0.9  # But the probability of wet after sprinkler is a little less
        self.graph["Holmes"].prior["ff"] = 0.2  # Holmes' grass is also be prone to cat-invasion

    def connect(self, parent, child):
        # Minimize emotional problems
        self.graph[parent].children[child] = self.graph[child]
        self.graph[child].parents[parent] = self.graph[parent]

    def estimate(self, x, wx):
        pass


class Node:
    def __init__(self, name):
        self.name = name
        self.parents = defaultdict(Node)
        self.children = defaultdict(Node)
        self.prior = defaultdict(float)
        self.p = None


if __name__ == '__main__':
    """
    bn = BayesianNetwork()

    for x in nodes:
        # Get all other nodes than x
        wx = nodes.copy()
        wx.remove(x)

        # Estimate P(x | Wx)
        bn.estimate(x, wx)
    """
    #####################
    # First some theory #
    #####################

    # Probability of B 'given' A: P(B|A) = P(A) * P(B) / P(A)
    # Probability of A and B: P(A&B) = P(A) * P(B)
    # Probability of A or B (mutually exclusive*): P(AvB) = P(A) + P(B)
    # Probability of A or B (not mutually exclusive): P(AvB) = P(A) + P(B) - P(A) * P(B)
    # Bayes' Theorem: P(Y|X) = P(X|Y) * P(Y) / P(X)
    # General Bayes': P(Y|X,E) = P(X|Y,E) * P(Y|E) / P(X|E)

    # *Mutually exclusive means they can not happen at the same time

    ##########################################
    # Hard-coding to learn Bayesian Networks #
    ##########################################

    # W - Watson
    # H - Holmes
    # R - Rain
    # S - Sprinkler

    # Now, lets calculate some probabilities:
    # P(H)
    # P(R|H)
    # P(S|H)
    # P(W|H)
    # P(R|W,H)
    # P(S|W,H)

    # Probability of Holmes' grass being wet?
    # We do not know P(H) directly, but can calculate it based on conditional probability:
    # P(H) = P(H|R,S) * P(R,S) + P(H|-R,-S) * P(-R,-S)
    p_H = 1.0 * (0.2 + 0.1 - (0.2 * 0.1)) + 0.1 * (0.8 + 0.9 - (0.8 * 0.9))
    print("Probability of Holmes' grass being wet:", p_H)  # 0.378

    # The same goes for Watson's grass being wet:
    # P(W) = P(W|R) * P(R) + P(W|-R) * P(-R)
    p_W = 1.0 * 0.2 + 0.2 * 0.8
    print("Probability of Watson's grass being wet:", p_W)  # 0.36
    # Which is slightly less because of not having a sprinkler

    # Probability of Rain given Holmes' grass is wet?
    # Here is where the Bayes' Rule come in:
    # P(R|H) = P(H|R) * P(R) / P(H)
    p_R_h = 1.0 * 0.2 / p_H
    print("Probability of Rain given Holmes' grass is wet:", p_R_h)  # 0.529

    # Probability of Rain given Watson's grass is wet?
    # Same rule apply here (Bayes' Theorem):
    # P(R|W) = P(W|R) * P(R) / P(W)
    p_R_w = 1.0 * 0.2 / p_W
    print("Probability of Rain given Watson's grass is wet:", p_R_w)  # 0.556
    # Which is slightly more because Watson's grass is not affected by a sprinkler, only the rain.

    # Probability of Sprinkler given Holmes' grass is wet?
    # P(S|H) = P(H|S) * P(S) / P(H)
    p_S_h = 0.9 * 0.1 / p_H
    print("Probability of Sprinkler given Holmes' grass is wet:", p_S_h)  # 0.238

    # Probability of Sprinkler given Watson's grass is wet?
    # Because of D-separation, this is simplified to just the probability of Sprinkler
    # P(S|W) = P(S)
    p_S_w = 0.1
    print("Probability of Sprinkler given Watson's grass is wet:", p_S_w)  # 0.1

    # What about Watson's grass being wet given Holmes' grass is wet? (BUT we do not know if its raining)
    # Because of D-separation between Watson and Holmes given Rain, we can simplify from this:
    # P(W|H) = P(W|H,R) * P(R|H) + P(W|H,-R) * P(-R|H)                                to this:
    # P(W|H) = P(W|R) * P(R|H) + P(W|-R) * P(-R|H)
    p_W_h = 1.0 * p_R_h + 0.2 * (1 - p_R_h)
    print("Probability of Watson's grass being wet given Holmes' grass is wet:", p_W_h)  # 0.623

    # Probability of Rain given Watson's AND Holmes' grass is wet?
    # Because the two probabilities are not mutually exclusive, we can just use disjunction:
    # P(AvB) = P(A) + P(B) - P(A) * P(B)
    # P(R|W,H) = P(R|W) + P(R|H) - P(R|W) * P(R|H)
    p_R_w_h = p_R_w + p_R_h - (p_R_w * p_R_h)
    print("Probability of Rain given Watson's and Holmes' grass is wet:", p_R_w_h)  # 0.791
    # Which is about the opposite of probability of wet grass given no rain (0.2)

    # Finally, the probability of Sprinkler given Watson's and Holmes' grass is wet?
    # P(S|W,H) = P(S|W) + P(S|H) - P(S|W) * P(S|H)
    p_S_w_h = p_S_w + p_S_h - (p_S_w * p_S_h)
    print("Probability of Sprinkler given Watson's and Holmes' grass is wet:", p_S_w_h)  # 0.314
