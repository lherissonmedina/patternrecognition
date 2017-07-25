from sklearn.neural_network import MLPClassifier


def build():
    return "MultilayerPerception", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1)
