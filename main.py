import numpy as np
import pandas as pd
import pickle
import global_variables as gv
from tqdm import tqdm
from DT import DecisionTree, classifyLoan, displayMetrics

def loadTree():
    '''Load a pre-built tree.'''

    # Load DecisionTree classifier
    with open('data/dt_classifier.pkl', 'rb') as file:
        clf = pickle.load(file)

    # return the classifier
    return clf

def buildTree():
    # load training data
    train = pd.read_csv('data/train.csv')

    # train Decision Tree classifier
    clf = DecisionTree(train, gv.TARGET, gv.ATTRIBUTES, True)

    # draw Decision Tree
    print('')
    clf.drawTree()

    # save the classifier
    # Open a file and use dump()
    with open('data/dt_classifier.pkl', 'wb') as file:
        pickle.dump(clf, file)

    print("Saved DecisionTree classifier as a pickle file.")

    # return the classifier
    return clf

def makePredictions(tree):
    '''Classify test data.'''

    
    # load test data
    test = pd.read_csv('data/test.csv')
    
    print('Classifying ', "{:,}".format(test.shape[0]), ' loans...')

    preds = []   # predicted class label
    probs = []   # probability of classification
    locs = []    # location of row that has no prediction

    for i in tqdm(range(0, test.shape[0])):
        predictions = classifyLoan(test.iloc[i], tree)
        if len(predictions) > 0:
            preds.append(predictions[0])
            probs.append(predictions[1])
        else:
            preds.append('Undefined')
            probs.append(-1)
            locs.append(i)

    # store predictions
    test['predicted'] = preds
    test['probability'] = probs

    # save predictions
    test.to_pickle('data/test_preds.pkl')
    print('Saved predictions as a pickle file.\n')

def evaluate():
    '''Display metrics table.'''

    print("\nModel's Evaluation\n")

    # load predictions file
    test_preds = pd.read_pickle('data/test_preds.pkl')
    data = test_preds[['loan_return', 'predicted']]

    # show metrics table
    displayMetrics(data)

def displayMenu():
    '''Display menu and prompt for user input.'''
    
    print("\nSelect one of the options below:")
    print("1) Build a new Decision Tree.")
    print("2) Display result of a pre-built tree.")
    print("3) Exit program.")

    try:
        # prompt for user input
        userInput = int(input("Please enter a number 1-3: "))

        # show error mesage if user entered invalid option
        if userInput not in range(1, 4):
            raise Exception()
    except:
        # show error message if user entered invalid option
        print("***ERROR: Invalid value entered. Please try again.")
    
    return userInput


# Begin main
if __name__ == "__main__":
    
    # show menu and get user input
    userInput = displayMenu()
    
    if userInput == 1:
        # re-build DecisionTree classifier
        clf = buildTree()

        # classify test data
        makePredictions(clf.getTree())

        # display metrics table
        evaluate()
    elif userInput == 2:    # Display structure of a pre-built tree
        # load a pre-trained classifier
        clf = loadTree()

        # draw Decision Tree
        print('')
        clf.drawTree()

        # display metrics table
        evaluate()
    # -- end if/else

    # exit the program
    print("\nGoodbye!\n")