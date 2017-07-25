import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score
import Classifiers
import json

for num_feature in range(4, 10):
    data = pd.read_csv("PhishingData1.csv")
    features = list(data.columns[:num_feature])
    Xs = data[features]
    y = data["Result"]

    scores = dict()
    num_tests = 100
    num_start = 25
    num_increment = 25
    total_tests = float(len(Classifiers.classifiers))

    for test, classifier in enumerate(Classifiers.classifiers):
        name, clf = classifier.build()
        scores[name] = dict()
        for size in range(num_start, 100, num_increment):
            test_size = float(size)/100
            acc, hamm, prec = 0.0, 0.0, 0.0
            for n in range(0, num_tests):
                X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=test_size)
                predictions = clf.fit(X_train, y_train).predict(X_test)
                acc += accuracy_score(y_test, predictions)
                hamm += hamming_loss(y_test, predictions)
                prec += precision_score(y_test, predictions, average='micro')

            scores[name][test_size] = {"Accuracy": acc/num_tests, "Hamming": hamm/num_tests, "Precision": prec/num_tests}

        print (test+1)/total_tests


    #print json.dumps(scores, indent=4)

    with open("dataHamming"+str(num_feature)+".csv", "wb") as d:
        for classifer in scores:
            d.write("," + classifer)

        d.write("\n")

        for size in range(num_start, 100, num_increment):
            test_size = float(size)/100

            d.write(str(test_size))

            for classifer in scores:
                acc = scores[classifer][test_size]["Accuracy"]
                d.write("," + str(acc))

            d.write("\n")







#with open("Models/svm.pickle", "wb") as fle:
    #    pickle.dump(clf, fle, protocol=pickle.HIGHEST_PROTOCOL)
