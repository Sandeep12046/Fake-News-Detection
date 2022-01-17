import pandas as pd

def convert_dataset(source, destination):

    data_TrainNews = pd.read_csv(source, sep = '\t')
    newData = pd.DataFrame(columns = ['Statement', 'Label'])

    # converting multiclass labels present in dataset to binary class labels
    for i , row in data_TrainNews.iterrows():
        currLabel = str(data_TrainNews.iloc[i, 1])
        currStatement = str(data_TrainNews.iloc[i, 2])
        if (currLabel == 'mostly-true' or currLabel == 'half-true' or currLabel == 'true'):
            newData = newData.append({'Statement': currStatement, 'Label': 'TRUE'}, ignore_index = True)
        else :
            newData = newData.append({'Statement': currStatement, 'Label': 'FALSE'}, ignore_index = True)

    newData.to_csv(destination, index = False, header = True)
