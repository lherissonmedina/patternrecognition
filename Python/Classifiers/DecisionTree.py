import os
from sklearn import tree

def build():
    return "DecisionTree", tree.DecisionTreeClassifier()


def make_tree(clf, features):

    class_names = clf.classes_.tolist()
    for x, y in enumerate(class_names):
        class_names[x] = str(y)

    tree.export_graphviz(clf, out_file='Classifiers/tree.dot', feature_names=features, class_names=class_names, filled=False)
    os.system('dot -Tpng Classifiers/tree.dot -o Classifiers/tree.png')
