import pickle
import DecisionTree, MLP, NearestNeighbors, RandomForest, SVM

classifiers = [DecisionTree, MLP, NearestNeighbors, RandomForest, SVM]


def save_classifier(clf, name):
    with open("Models/"+name+".pickle", "wb") as fle:
        pickle.dump(clf, fle, protocol=pickle.HIGHEST_PROTOCOL)

def train(clf, sample, result):
    clf.fit(sample, result)
    return clf
