"""
@File: regression-gradient.py
@Author: luyufan
@Date: 2020/4/10
@Desc: Implementation of Regression without toolkit.
@Ref: Homework one for: http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html
"""

"""
@Data Desc: The train date includes many attibutes for weather every hour in a day,
so one record for a day consists or a matrix which rows represent an attribute and cols represent hours from 0 to 23.
"""

"""
@Analysis: Note that our final task is giving sequential 9 hours data to predict the 10-th data, so it is necceary to split 
all the training data into several mini data element with sequential 9 hours data and the 10-th as the target. Meanwhile, 
The whole training data should be splited to validation data.
BTW, It is useful to training with RNN in this kind of data processing.
"""

import numpy as np,pandas as pd
import torch.utils.tensorboard as tensorboard

def read_data(filename = None):
    """
    reshape the raw data to a huge matrix
    ignore the date.
    """
    raw_data = None
    try:
        raw_data = pd.read_csv(filename)
    except IOError() as e:
        print("Can not open file.",e.errno)
        return None
    return raw_data

def extract_data(data):
    """
    x-axis: 18-attributes.
    y-axis: each hour on each day.
    replace NR to 0 in RIANFALL
    """
    assert data is not None
    attributes_num = 18
    data_np = None
    for i in range(0,data.shape[0],attributes_num):
        if data_np is None:
            data_np = np.array(data.iloc[i:i + attributes_num,2:])
        else:
            everyday_data = np.array(data.iloc[i:i + attributes_num,2:])
            data_np = np.column_stack((data_np,everyday_data))
    data_np = np.transpose(data_np)
    # replace NR to 0 in RIANFALL
    data_np[data_np=="NR"] = 0
    data_np = data_np.astype(np.float)
    return data_np

def generate_data_pairs(data,start_time_index,get_target=True):
    """
    make (0~9 feature_data,10-th target pairs)
    """
    if get_target:
        assert start_time_index + 9 <= data.shape[0]-1
        feature_data = data[start_time_index:start_time_index+9,:]
        target_data = data[start_time_index+10,:]
        return feature_data,target_data
    else:
        feature_data = data[start_time_index:start_time_index+9,:]
        return feature_data

def split_datasets_index_list(total_length):
    """
    split the whole data into train set and validation set
    """
    whole_index_list = np.asarray(range(0,total_length-10))
    # 80% ~ 20% validation
    train_set_list = whole_index_list[:int(len(whole_index_list) * 0.8)]
    validation_set_list = whole_index_list[len(train_set_list):]
    return train_set_list,validation_set_list

def normalize_dateset(data,data_mean = None,data_std = None):
    """
    normalize the whole raw data.
    """
    if data_mean is None:
        data_mean = np.mean(data,axis = 0)
    if data_std is None:
        data_std = np.std(data,axis = 0) + 1e-8
    return data_mean,data_std,(data - data_mean.reshape(-1,18)) / (data_std.reshape(-1,18))

def shuffle_list(list_data):
    np.random.shuffle(list_data)

def train(normalized_dataset,trainset_list,validation_list = None):
    """
    y_hat = AX + b
    a naive implementation of SGD. without any improving method.
    """
    writer = tensorboard.SummaryWriter(log_dir="./tensorboard",flush_secs=0.1)
    # composed all weights into one big matrix weight.
    composed_weight = np.random.randn(18, (18 * 9) + 1)
    # set learning rate
    lr = 0.001
    global_step = 0
    for epoch in range(100):
        shuffle_list(trainset_list)
        for i,data_index in enumerate(trainset_list):
            feature,targtet = generate_data_pairs(normalized_dataset,data_index)
            feature = feature.transpose()
            targtet = targtet.transpose()
            feature = feature.reshape(-1,1)
            targtet = targtet.reshape(-1,1)

            # compose new input considering bias
            composed_x = np.row_stack((feature,np.array(1)))

            # compute output
            y_hat = np.dot(composed_weight,composed_x)

            # update gradient: think? why?
            gradient = 2 * np.dot(y_hat-targtet,composed_x.transpose())
            composed_weight -= lr * gradient

            # record and vizualize
            global_step+=1

            # get loss
            loss = np.mean(np.sum(np.power(y_hat - targtet, 2)))
            writer.add_scalar("loss/training loss",loss,global_step)

            if global_step % 1000 == 0:
                print("Epoch:{ep}. ".format(ep = epoch),"Iteration:{iter}. ".format(iter=i),"Loss:{lo}".format(lo=loss))
                # validation mse
                loss = 0
                if validation_list is not None:
                    for j,index in enumerate(validation_list):
                        feature,targtet = generate_data_pairs(normalized_dataset,index)
                        feature = feature.transpose()
                        targtet = targtet.transpose()
                        feature = feature.reshape(-1,1)
                        targtet = targtet.reshape(-1,1)
                        y_hat = np.dot(composed_weight,composed_x)
                        loss += np.mean(np.sum(np.power(y_hat - targtet, 2)))
                    loss /= (i+1)
                    writer.add_scalar("loss/validation loss",loss,global_step)

    np.save("./composed_weight.npy",composed_weight)
    np.save("./data_mean.npy",data_mean)
    np.save("./data_std.npy",data_std)

def prediction(normalized_dataset,composed_weight,data_mean,data_std):
   
    for index in range(0,len(normalized_dataset),9):
        feature = generate_data_pairs(normalized_dataset,index,get_target=False)
        feature = feature.transpose()
        feature = feature.reshape(-1,1)
        composed_x = np.row_stack((feature,np.array(1)))
        y_hat = np.dot(composed_weight,composed_x)
        y_hat = y_hat * data_std.reshape(-1,1) + data_mean.reshape(-1,1)

        print("Prediction PM2.5 for ID{index}".format(index=index / 9),y_hat[10])

if __name__ == "__main__":

    # # read raw data
    # raw_data = read_data("./data/train.csv")
    # # extract data into 5760 * 18 matrix
    # extracted_data = extract_data(raw_data)
    # # normalize data
    # data_mean,data_std,normalized_dataset= normalize_dateset(extracted_data)
    # # split data set
    # trainset_list,validation_list = split_datasets_index_list(normalized_dataset.shape[0])

    # train(normalized_dataset,trainset_list,validation_list)

    # read raw data
    raw_data = read_data("./data/test.csv")
    # extract data into 2160 * 18 matrix
    extracted_data = extract_data(raw_data)
    # normalize data
    # load weights
    composed_weight = np.load("./composed_weight.npy")
    data_mean,data_std = np.load("./data_mean.npy"),np.load("./data_std.npy")
    _,_,normalized_dataset= normalize_dateset(extracted_data,data_mean,data_std)
    prediction(normalized_dataset,composed_weight,data_mean,data_std)
