############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Data Mining
# Lesson: Naive Bayes Classification

# Citation: 
# PEREIRA, V. (2018). Project: Naive Bayes Classification, File: Python-DM-Classification-07-NB.py, GitHub repository: <https://github.com/Valdecy/Naive Bayes Classification>

############################################################################

# Installing Required Libraries
import pandas as pd
import numpy  as np

# Function: Prediction           
def naive_bayes_prediction(nb_model, Test_data): 
    Test_data = Test_data.reset_index(drop=True)
    ydata = pd.DataFrame(index=range(0, Test_data.shape[0]), columns=["Prediction"])
    data  = pd.concat([ydata, Test_data], axis = 1)
    data = data.applymap(str)
    for i in range(0, data.shape[0]):
        for k in range(0, nb_model[1].shape[0]):
            likelihood = 1
            for j in range(1, data.shape[1]):
                idx = nb_model[0].index[nb_model[0]['Categories'] == data.iloc[i,j]].tolist()[0]
                likelihood = likelihood*nb_model[0].iloc[idx, k + 3]
            posteriori = ((likelihood*nb_model[1].iloc[k,2]))   
            print("p(", nb_model[1].iloc[0,0],   "{", nb_model[1].iloc[k,1], "} │ X_", i+1, ")", "=", round(posteriori, 4))
            if (k==0):
                data.iloc[i,0] = nb_model[1].iloc[k,1]
                temp = posteriori
            elif(temp < posteriori):
                data.iloc[i,0] = nb_model[1].iloc[k,1]
                temp = posteriori
    return data

# Function: Naive Bayes Classication
def naive_bayes_classification(Xdata, ydata, laplacian_correction = True):
    label_name = ydata.name
    ydata = pd.DataFrame(ydata.values.reshape((ydata.shape[0], 1)))
    dataset = pd.concat([ydata, Xdata], axis = 1)
    dataset = dataset.applymap(str)
    
    size = 0
    category = []
    attribute =[]
    unique_categories = []
    unique_attributes = []
    
    for j in range(0, dataset.shape[1]): 
        for i in range(0, dataset.shape[0]):
            token_1 = dataset.iloc[i, j]
            token_2 = list(dataset)[j]
            if not token_1 in category:
                category.append(token_1)
                attribute.append(token_2)
                if j >0:
                    size = size + 1
        unique_categories.append(category)
        unique_attributes.append(attribute)
        category = [] 
        attribute = []
        
    sequence_1 = []
    for element in unique_categories:
        if (element != unique_categories[0]):
            sequence_1 = sequence_1 + element

    sequence_2 = []
    for element in unique_attributes:
        if (element != unique_attributes[0]):
            sequence_2 = sequence_2 + element
   
    probability_table = pd.DataFrame(np.zeros((size, 3)), columns = ["Attributes", "Categories", "Frequency"])
    for i in range (0, len(unique_categories[0])):
        name = unique_categories[0][i]
        probability_table[name] = 0    
        
    for i in range(0, len(sequence_1)):
        probability_table.iloc[i, 0] = sequence_2[i]
        probability_table.iloc[i, 1] = sequence_1[i]
       
    for i in range(1, dataset.shape[1]):        

        contigency_table = pd.crosstab(dataset.iloc[:,0], dataset.iloc[:,i], margins = False)
        category = list(contigency_table)
        for j in range(3, probability_table.shape[1]):
            label = list(probability_table)[j]
            for k in range(0, len(category)):
                if (laplacian_correction == False):
                    numerator = contigency_table.loc[label,category[k]]
                    divisor = contigency_table.loc[label,:].sum()
                    idx = probability_table.index[probability_table['Categories'] == category[k]].tolist()[0]
                    probability_table.loc[idx, label] = numerator/divisor
                    probability_table.loc[idx, "Frequency"] = probability_table.loc[idx, "Frequency"] + contigency_table.loc[label,category[k]]/contigency_table.values.sum()
                else:
                    numerator = contigency_table.loc[label,category[k]] + 1
                    divisor = contigency_table.loc[label,:].sum() + len(category)
                    idx = probability_table.index[probability_table['Categories'] == category[k]].tolist()[0]
                    probability_table.loc[idx, label] = numerator/divisor
                    probability_table.loc[idx, "Frequency"] = probability_table.loc[idx, "Frequency"] + contigency_table.loc[label,category[k]]/contigency_table.values.sum()
    
    labels_all = pd.crosstab(dataset.iloc[:,0], dataset.iloc[:,0], margins = False)
    probability_labels = pd.DataFrame(np.zeros((len(unique_categories[0]), 3)), columns = ["Attributes", "Categories", "Probability"])
    for i in range(0,probability_labels.shape[0]):
        if (laplacian_correction == False):
            probability_labels.iloc[i,0] = label_name
            probability_labels.iloc[i,1] = unique_categories[0][i]
            probability_labels.iloc[i,2] = labels_all.iloc[i,i]/labels_all.values.sum()
        else:
            probability_labels.iloc[i,0] = label_name
            probability_labels.iloc[i,1] = unique_categories[0][i]
            probability_labels.iloc[i,2] = (labels_all.iloc[i,i] + 1)/(labels_all.values.sum() + len(unique_categories[0]))

    return [probability_table, probability_labels]

######################## Part 1 - Usage ####################################

# Trainning the Classifier
trainning = pd.read_csv('Python-DM-Classification-07-NBa.csv', sep = ';')
X = trainning.iloc[:, 0:4]
y = trainning.iloc[:, 4]
nb_model = naive_bayes_classification(Xdata = X, ydata = y, laplacian_correction = False)

# Prediction
test =  pd.read_csv('Python-DM-Classification-07-NBb.csv', sep = ';')
Z = test.iloc[:, 0:4]
pred = naive_bayes_prediction(nb_model, Z)

########################## End of Code #####################################
