from IPython.display import display
from sklearn.utils import shuffle
from tqdm import tqdm
import pandas as pd
import numpy as np
import global_variables as gv
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def save_data(dataIn, filepath):
    '''Save dataframe to a csv file with a progress bar.'''

    # save data using multiple chunks (chunksize = 100)
    ixs = np.array_split(dataIn.index, 100)

    for i, subset in tqdm(enumerate(ixs)):
        if i == 0:
            dataIn.loc[subset].to_csv(filepath, mode='w', index=False)
        else:
            dataIn.loc[subset].to_csv(filepath, header=None, mode='a', index=False)

def annotate(chart):
    # place value label above each bar
    for p in chart.patches:
        chart.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                       textcoords='offset points')

def countplot(data, attribute, xlabel='Loan Return'):
    # setting the dimensions of the plot
    fig, ax = plt.subplots(figsize=(8, 4))

    # plot bar plot
    chart = sns.countplot(x=attribute, data=data)
    annotate(chart)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.title('Data Distribution per ' + xlabel, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel('Number of Loans', fontsize=12, fontweight='bold')
    plt.xticks(rotation=90)
    plt.show()

# Begin main
if __name__ == "__main__":
    # load pre-processed data
    data = pd.read_pickle('data/processed_data.pkl')

    # get a list of loan_return class labels
    loan_return = list(data[gv.TARGET].unique())

    # get a count for each loan_return class label
    print('Count number of examples for each class...')
    label_count = [data[data[gv.TARGET] == label].shape[0] for label in tqdm(loan_return)]

    # number of training examples = 70% of label count for each class
    print('Compute number of examples for train dataset...')
    train_count = [round(0.7*count) for count in tqdm(label_count)]

    print('Before shuffling data...')
    display(data.head())

    # shuffle data
    data = shuffle(data, random_state=42)
    data.reset_index(drop=True, inplace=True)

    print('\nAfter shuffling data...')
    display(data.head())

    # stratified sampling without replacement
    # get the first n examples from each class on shuffled data
    print('Perform stratified sampling without replacement...')
    train_list = []
    test_list = []
    for i, count in enumerate(tqdm(train_count)):
        # setlect first 70% examples for the given class as train data
        train_list.append(data[data[gv.TARGET] == loan_return[i]][:count])   
        
        # select last 30% examples for the given class as test data
        test_list.append(data[data[gv.TARGET] == loan_return[i]][count:])

    # concatenate dataframes for test/train
    print('Prepare train dataset...')
    train = pd.concat(tqdm(train_list))
    print('Prepare test dataset...')
    test = pd.concat(tqdm(test_list))
    
    # attributes to be saved
    cols = gv.ATTRIBUTES
    cols.append(gv.TARGET)

    # save train/test dataset
    print('Save train dataset...')
    train[cols].to_csv('data/train.csv', index=False)
    #save_data(train, 'data/train.csv')
    print('Save test dataset...')
    #save_data(test, 'data/test.csv')
    test[cols].to_csv('data/test.csv', index=False)

    # visualize train dataset
    #print('Visualize the distribution of train dataset...')
    #countplot(train, gv.TARGET)