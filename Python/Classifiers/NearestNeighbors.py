from sklearn.neighbors import KNeighborsClassifier


def build():
    return "NearestNeighbors", KNeighborsClassifier(n_neighbors=10)
