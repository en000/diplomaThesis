from scipy.io import loadmat
import numpy as np
import os
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LogisticRegression
from operator import itemgetter
from sklearn.ensemble import RandomForestClassifier

def mrmr_selection(data, relevance_scores, redunduncy_scores, n):

    # Normalize scores
    relevance_scores = (relevance_scores-np.nanmin(relevance_scores))/(np.nanmax(relevance_scores)-np.nanmin(relevance_scores))
    redunduncy_scores = MinMaxScaler().fit_transform(redunduncy_scores)

    selected_features = []

    # Select feature with best score
    max_relevance_index = np.argmax(relevance_scores)
    selected_features.append(max_relevance_index)
    relevance_scores[max_relevance_index]=0

    for i in range(n-1):

        # For each feature get the redundancies to the already selected features
        # Get the mean average distance for each feature
        means = []
        suma = []
        for row in range(len(redunduncy_scores)):

            redunduncies = redunduncy_scores[row,[selected_features]]

            avr = np.mean(redunduncies)

            means.append(avr)

        # Pair scores with feature indices
        scores = relevance_scores - means
        tuples = []
        for i, s in enumerate(scores):
            tuples.append((i, s))

        # Sort by score
        sortedScores = sorted(tuples,key=itemgetter(1))[::-1]

        # Choose feature with max score that hasn't been selected
        i = 0
        bestIndex, bestScore = sortedScores[i]
        while bestIndex in selected_features:
            i+=1
            bestIndex, bestScore = sortedScores[i]

        selected_features.append(bestIndex)

    return selected_features

def main():

    # Load data
    # In folder data
    os.chdir('data')

    # Choose database
    data = loadmat('nci9.mat')

    le = LabelEncoder()
    data = np.append(data['X'], data['Y'], axis=1)
    le.fit(np.unique(data))
    data = np.array([le.transform(samp) for samp in data])

    # Shuffle instances
    np.random.shuffle(data)

    XFull = np.array(data[:, :-1])
    yFull = data[:, -1]

    # Split into training and testing data
    X = XFull[:40]
    y = yFull[:40].flatten()
    
    testX = XFull[41:]
    testY = yFull[41:]

    accuraciesMrmr = []
    accuraciesFiltering = []

    # Code for getting Random Forest importance Estimates
    scores = RandomForestClassifier().fit(X, y).feature_importances_
        
    toup = []
    for i, x in enumerate(scores):
        toup.append((i, x))
    sort = sorted(toup,key=itemgetter(1), reverse=True)

    # Code for getting Mutual Information redundancy estimates
    # similarities = np.zeros((len(X[0]),len(X[0])))
    # for i in range(len(X[0])):
    #     print("i="+str(i))
    #     for j in range(i, len(X[0])):
    #         # Mutual info to features
    #         similarities[i][j] = mutual_info_score(X[:,i], X[:,j])
    #         similarities[j][i] = similarities[i][j]

    # Code for getting Chi2 redundancy estimates
    similarities = np.zeros((len(X[0]),len(X[0])))
    for i in range(len(X[0])):
        print("i="+str(i))
        c = chi2(X, X[:,i])
        similarities[i] = (sum(c)/len(c))

    # Set values of k
    ks = [1]
    for i in range(8):
        ks.append(ks[-1]*2)

    for k in ks:

        # Filter method for Random Forest importance estimates
        selectK = []
        for a in range(0,k):
            i, x = sort[a]
            selectK.append(i)
        
        X_selectedKBest = XFull[:, selectK]
        trainXKBest = X_selectedKBest[:40]
        testXKBest = X_selectedKBest[41:]

        # Filter method for Chi2 importance estimates
        # selectedKBest = SelectKBest(chi2, k=k).fit(X, y).get_support()
        # trainXKBest = X[:, selectedKBest]
        # testXKBest = testX[:, selectedKBest]
        
        # Relevances
        feature_importances = scores

        # Code for Chi2 importance estimates
        # feature_importances = sum(chi2(X, y))

        selectedMrmr = mrmr_selection(X, feature_importances, np.array(similarities), k)

        print("Selected features indexes:")
        print(selectedMrmr)

        model = LogisticRegression().fit(X[:, selectedMrmr], y)
        y_predMrmr = model.predict(testX[:, selectedMrmr])

        accMrmr=accuracy_score(testY, y_predMrmr)

        print("Accuracy for mrmr: ", accMrmr)
        accuraciesMrmr.append(accMrmr)

        lg = LogisticRegression().fit(trainXKBest, y)
        y_prednotmy = lg.predict(testXKBest)

        accFilter=accuracy_score(testY, y_prednotmy)

        print("Accuracy for k best: ", accFilter)
        accuraciesFiltering.append(accFilter)

    print(accuraciesMrmr)
    print(accuraciesFiltering)

    # Show graph and results

    x = np.array(range(9))
    X_Y_Spline = make_interp_spline(range(9), accuraciesMrmr)
    X_ = np.linspace(x.min(), x.max())
    Y_ = X_Y_Spline(X_)
    
    plt.plot(X_, Y_, color='g', label='mrmr', zorder=1)
    plt.scatter(x, accuraciesMrmr, color="blue", zorder=2)


    X_Y_Spline = make_interp_spline(range(9), accuraciesFiltering)
    X_ = np.linspace(x.min(), x.max())
    Y_ = X_Y_Spline(X_)
    
    plt.plot(X_, Y_, color='r', label='kbest', zorder=1)
    plt.scatter(x, accuraciesFiltering, color="blue", zorder=2)
    plt.xticks(range(9), ks)

    plt.xlabel("k nci")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()