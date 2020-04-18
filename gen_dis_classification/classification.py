"""
@File: classification.py
@Author: luyufan
@Date: 2020/4/17
@Desc: Implementation of Classification without toolkit.
@Ref: Homework one for: http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html
"""

"""
@Data Desc: This Data set comes from UCI ML website http://archive.ics.uci.edu/ml/datasets/Census+Income, designed to binary classification.
Description: It contains weighted census data extracted from the 1994 and 1995 Current
Population Surveys conducted by the U.S. Census Bureau. The data contains 41 demographic and employment related variables.
--------------------------------
age ----> int64 ----> 91  unique values  √
class of worker ----> object ----> 9  unique values √
detailed industry recode ----> int64 ----> 52  unique values  
detailed occupation recode ----> int64 ----> 47  unique values
education ----> object ----> 17  unique values   √
wage per hour ----> int64 ----> 761  unique values √
enroll in edu inst last wk ----> object ----> 3  unique values  √
marital stat ----> object ----> 7  unique values √
major industry code ----> object ----> 24  unique values  
major occupation code ----> object ----> 15  unique values √
race ----> object ----> 5  unique values √
hispanic origin ----> object ----> 10  unique values √
sex ----> object ----> 2  unique values √
member of a labor union ----> object ----> 3  unique values
reason for unemployment ----> object ----> 6  unique values
full or part time employment stat ----> object ----> 8  unique values √
capital gains ----> int64 ----> 126  unique values
capital losses ----> int64 ----> 105  unique values
dividends from stocks ----> int64 ----> 1047  unique values
tax filer stat ----> object ----> 6  unique values √
region of previous residence ----> object ----> 6  unique values
state of previous residence ----> object ----> 51  unique values
detailed household and family stat ----> object ----> 36  unique values
detailed household summary in household ----> object ----> 8  unique values  √
migration code-change in msa ----> object ----> 10  unique values
migration code-change in reg ----> object ----> 9  unique values
migration code-move within reg ----> object ----> 10  unique values
live in this house 1 year ago ----> object ----> 3  unique values
migration prev res in sunbelt ----> object ----> 4  unique values
num persons worked for employer ----> int64 ----> 7  unique values √
family members under 18 ----> object ----> 5  unique values √
country of birth father ----> object ----> 43  unique values
country of birth mother ----> object ----> 43  unique values
country of birth self ----> object ----> 43  unique values
citizenship ----> object ----> 5  unique values
own business or self employed ----> int64 ----> 3  unique values √
fill inc questionnaire for veteran's admin ----> object ----> 3  unique values
veterans benefits ----> int64 ----> 3  unique values √
weeks worked in year ----> int64 ----> 53  unique values √
year ----> int64 ----> 2  unique values
y ----> object ----> 2  unique values
--------------------------------
w.r.t the raw dataset, some attributes are not suitable for the classification task, so it is necessary to wash the data.
note that either generation model or discrimination model needs num-data instead of string used in desecion tree.
"""

"""
@Analysis:
For classification, generation model or discriminal model are different ways.
and if the attribute is discrete type it will be encoded in one-hot,
on the contrast, the continuous will be processed as normal.
some attributes may have so many dimensions so it is not a good candidate or it need to be trimed.
so this file will implement the above methods.

@Result:
Generation Model Acc: 76%
Discrimination Acc: 81.5%
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_save_data(csv_file_path = None):
    """
    process the raw data to generate train matrix with attributes selected.
    matrix shape: rows * (selected attr bits)
    """
    raw_train_data = pd.read_csv("./data/train.csv",index_col="id")
    rows,_ = raw_train_data.shape
    attr_width_dict = {
        "age":1,
        "class of worker":9,
        "education":17,
        "wage per hour":1,
        "enroll in edu inst last wk":3,
        "marital stat":7,
        "major occupation code":15,
        "race":5,
        "hispanic origin":10,
        "sex":1,  # binary
        "full or part time employment stat":8,
        "tax filer stat":6,
        "detailed household summary in household":8,
        "num persons worked for employer":7,
        "family members under 18":5,
        "own business or self employed":1,
        "veterans benefits":1,
        "weeks worked in year":1,
        "y":1,   # label
    }
    attr_unique_dict = {}
    instance_length = 0
    for l in attr_width_dict.values():
        instance_length += l
    
    coded_dataset = np.zeros((rows,instance_length),dtype = float)
    for row in range(rows):
        col_point = 0       # position in each row
        for attr,width in attr_width_dict.items():       # encode each attribute by rows
            if width != 1:                               # encode object attributes
                if attr not in attr_unique_dict.keys():  
                    attr_unique_dict[attr] = sorted(raw_train_data.loc[:,attr].unique())
                value_index = attr_unique_dict[attr].index(raw_train_data.loc[row,attr])
                coded_dataset[row,col_point+value_index] = 1  # encoding
            elif attr == "sex":                               # encode set attribute
                coded_dataset[row,col_point] = 1 if raw_train_data.loc[row,attr] == "Male" else 0
            elif attr == "y":
                coded_dataset[row,col_point] = 0 if (raw_train_data.loc[row,attr].find("-")!=-1) else 1
            else:
                coded_dataset[row,col_point] = raw_train_data.loc[row,attr]
            col_point += width
    np.save("./data/coded_data.npy",coded_dataset)

def read_encoded_data(data_path = None):
    """
    load dataset
    """
    data = None
    try:
        data = np.load(data_path)
    except IOError as e:
        print(e)
    return data

def normalize(data):
    """
    normalize the dataset
    mean = 1/N∑x
    std = sqrt(1/N∑(x - mean)**2 + delta) = sqrt(1/N∑x**2 - mean**2)
    """
    mean = np.mean(data,axis = 0).reshape(1,-1)
    std = np.std(data,axis = 0).reshape(1,-1) + 1e-8
    data = (data - mean) / std
    return data,mean,data

def sigmoid(z):
    """
    Sigmoid function can be used to calculate probability for label 1.
    To avoid overflow, minimum/maximum output value is set.
    """
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def accuracy(y_hat,y):
    """
    calculate acc.
    """
    acc = np.mean(1 - np.abs(y.reshape(-1,1) - y_hat.reshape(-1,1)))
    return acc

def compute_cross_entropy_loss(y_hat,y_label):
    """
    cross entropy: 
    """
    cross_entropy_loss = -(
        np.matmul(y_label.T,np.log(y_hat)) +
        np.matmul((1-y_label).T,np.log(1 - y_hat))
        )
    return cross_entropy_loss

def compute_gradient(x_feature,y_pred,y_label):
    """
    compute the gradients
    """
    pred_error = y_label - y_pred
    w_grad = pred_error.T * x_feature.T
    w_grad = -np.sum(w_grad,axis = 1).reshape(-1,1)
    b_grad = -np.sum(pred_error)
    return w_grad,b_grad

def generation_model():
    """
    use generation model to classify
    u1,u2,Σ
    """
    data = read_encoded_data("./data/coded_data.npy")
    # It needs mean and std for testing.
    normalized_data,mean,std = normalize(data[:,:-1])
    pos_class_pos = np.where(data[:,-1] == 1)
    neg_class_pos =  np.where(data[:,-1] == 0)
    pos_instances_feature = normalized_data[pos_class_pos]
    neg_instances_feature = normalized_data[neg_class_pos]
    u1 = np.mean(pos_instances_feature,axis =0)
    Σ1 = np.cov(pos_instances_feature.T)
    u2 = np.mean(neg_instances_feature,axis =0)
    Σ2 = np.cov(neg_instances_feature.T)
    ratio1  = pos_instances_feature.shape[0] / (pos_instances_feature.shape[0] + neg_instances_feature.shape[0])
    ratio2  = neg_instances_feature.shape[0] / (pos_instances_feature.shape[0] + neg_instances_feature.shape[0])
    Σ = ratio1 * Σ1 + ratio2 * Σ2
    
    # compute inverse of Σ in case of linear
    u,s,v = np.linalg.svd(Σ, full_matrices=False)
    Σ_inv = np.matmul(v.T * 1 / s, u.T)
    
    # compute w and b for σ(z)
    # (u1 - u2) is col-vector so it does not need transpose
    w = np.matmul((u1-u2).transpose(),Σ_inv)
    b = (-0.5) * np.matmul(np.matmul(u1.transpose(),Σ_inv),u1) 
    + (0.5) * np.matmul(np.matmul(u2.transpose(),Σ_inv),u2) 
    + np.log(float(neg_instances_feature.shape[0]) / pos_instances_feature.shape[0])
   
    # compute predictions
    trained_features_z = np.matmul(normalized_data,w) + b
    predictions = sigmoid(trained_features_z)
    acc = accuracy(predictions,data[:,-1])
    print(acc)

def discriminal_model():
    """
    use gradient-descent to update w and b for discriminal model
    """
    data = read_encoded_data("./data/coded_data.npy")
    # It needs mean and std for testing.
    normalized_data,mean,std = normalize(data[:,:-1])

    w = np.random.rand(normalized_data.shape[1],1)
    b = 0
    lr = 0.001
    
    loss_value = []
    train_acc_value = []
    loss_step = 0
    acc_step = 0
    plt.figure()
    for epoch in range(100):
        shuffle_index = np.array(range(normalized_data.shape[0]))
        np.random.shuffle(shuffle_index)
        batch_size = 60
        for i in range(0,len(shuffle_index),batch_size):
            if len(shuffle_index) - i < batch_size:
                break
            
            # data batch
            x_feature = normalized_data[[shuffle_index[i+j] for j in range(batch_size)]]
            y_label = data[[shuffle_index[i+j] for j in range(batch_size)],-1].reshape(-1,1)
            trained_features_z = np.matmul(x_feature,w) + b
            y_pred = sigmoid(trained_features_z)

            # simple update
            loss = compute_cross_entropy_loss(y_pred,y_label)
            w_gradient,b_gradient =  compute_gradient(x_feature,y_pred,y_label)
            w -= lr * w_gradient
            b -= lr * b_gradient
            print("Entrophy Loss",loss[0])

            # statistic.
            loss_value.append(loss[0])
            if loss_step % 5000 == 0:
               
                # training acc:
                x_feature = normalized_data[0:1000]
                y_label = data[0:1000,-1]
                trained_features_z = np.matmul(x_feature,w) + b
                y_pred = sigmoid(trained_features_z)
                acc = accuracy(y_pred,y_label)
                train_acc_value.append(acc * 100)
                print("Train acc:",acc)
               
                # plot
                plt.plot(list(range(loss_step + 1)),loss_value)
                plt.plot(np.array((range(acc_step + 1))) * 5000, train_acc_value)
                plt.show()

                acc_step += 1

            loss_step += 1

if __name__ == "__main__":
    # generation_model()
    discriminal_model()
