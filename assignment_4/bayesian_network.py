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


class BayesianNetwork:
    def __init__(self):
        self.nodes = None

    def connect(self, parent, child):
        # Minimize emotional problems
        parent.children.append(child)
        child.parents.append(parent)


class Probability:
    def __init__(self):
        self.Watson = dict()
        self.Holmes = dict()

        # Probability of moisturized grass given (rain, sprinkler)
        self.Watson['tt'] = 0.9
        self.Watson['ff'] = 0.05
        self.Watson['tf'] = 0.9
        self.Watson['ft'] = 0.05

        self.Holmes['tt'] = 0.99
        self.Holmes['ff'] = 0.05
        self.Holmes['tf'] = 0.9
        self.Holmes['ft'] = 0.6

    def is_watson_moist(self, key):
        return self.Watson[key]

    def is_holmes_moist(self, key):
        return self.Holmes[key]


class Node:
    def __init__(self, name, parents, children):
        self.name = name
        self.parents = parents
        self.children = children



