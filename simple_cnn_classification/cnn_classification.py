"""
@File: cnn_classification.py
@Author: luyufan
@Date: 2020/4/19
@Desc: Use cnn to classify images based on Pytorch
@Ref: Homework one for: http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html
"""

"""
@Data Desc: This food dataset is crawled from network which contains 11 different kinds of food images, so this task is to classify them.
@Analysis:
Data Processing
Normalization, Augmentation

@Result: Bad result, Only 43% acc in validation after 20 epochs.
"""

import torch as torch
import torch.nn as nn
import torch.optim as optim
import tqdm,os,json
import PIL.Image as PILImage
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms


class FoodDataSet(Dataset):
    """
    Dataset of Food Images
    """
    def __init__(self,data_root,mode,transform):
        super(FoodDataSet,self).__init__()
        self.data_root = data_root
        self.transform = transform
        self.mode = mode
        assert mode in ["train","eval","test"]
        if mode == "train":
            self.image_folder = "training"
        elif mode == "eval":
            self.image_folder = "validation"
        else:
            self.image_folder = "testing"
        self.img_name_list = os.listdir(os.path.join(self.data_root,self.image_folder))
    def __len__(self):
        return len(self.img_name_list)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image_name = self.img_name_list[index]
        if type(image_name) == list:
            # TODO
            print("The index is a list")
        else:
            image_path = os.path.join(self.data_root,self.image_folder,image_name)
            image = PILImage.open(image_path)
            image = self.transform(image)
            if self.mode == "test":
                return image
            else:
                lable = int(image_name.split("_")[0])
                return image,lable

class FoodClassificationModel(nn.Module):
    """
    Simple Image Classification model.
    """
    def __init__(self):

        super(FoodClassificationModel,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=2,padding = 2, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3,stride=2,padding = 1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3,stride=2,padding = 1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(num_features=256)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding = 1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding = 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(num_features=512)
        self.conv6 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=2,padding = 1, bias=False)
        self.conv7 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=2,padding = 1, bias=False)
        self.linear = nn.Linear(in_features=512 * 4,out_features=11,bias=True)

        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = x.view(-1,512 * 4)
        output = self.linear(x)
        return output

def evaluation(model,transform,device,test=False):
    """
    Testing or evaluating interface.
    """
    if test:
        # TODO same as eval
        pass
    else:
        food_eval_data_set = FoodDataSet("./data",mode="eval",transform=transform)
        food_eval_dataloader = DataLoader(food_eval_data_set,batch_size=16,num_workers=0,shuffle=True)
        # statistics the confusing matrix.
        confuse_matrix_dict = {0:{},1:{},2:{},3:{},4:{},5:{},6:{},7:{},8:{},9:{},10:{}}
        acc_num = 0
        
        for i,data in enumerate(tqdm.tqdm(food_eval_dataloader,desc="Evaluation:")):
            with torch.no_grad():
                image_batch_tensor,label_batch_tensor = data
                image_batch_tensor = image_batch_tensor.to(device)
                label_batch_tensor = label_batch_tensor.to(device)
                output = model(image_batch_tensor)
                class_preds = torch.argmax(output,dim=1)
                for label,pred in zip(label_batch_tensor.cpu(),class_preds.cpu()):
                    label = label.item()
                    pred = pred.item()
                    if label == pred:
                        acc_num += 1
                    if pred not in confuse_matrix_dict[label].keys():
                        confuse_matrix_dict[label][pred] = 1
                    else:
                        confuse_matrix_dict[label][pred] += 1
        beautiful_format = json.dumps(confuse_matrix_dict, indent=4, ensure_ascii=False)
        print("Total acc:",acc_num / len(food_eval_data_set))
        print("Confusing Matrix:",beautiful_format)

def train(transform,device):
    """
    Training interface
    """
    food_train_data_set = FoodDataSet("./data",mode="train",transform=transform)
    food_train_dataloader = DataLoader(food_train_data_set,batch_size=16,num_workers=0,shuffle=True)

    food_classification_model = FoodClassificationModel().to(device)
    food_classification_model.train()
    # print(food_classification_model.state_dict())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(food_classification_model.parameters(),lr=0.001)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5000,10000],gamma=0.5)
    
    global_step = 0
    for epoch in range(20):
        for i,data in enumerate(tqdm.tqdm(food_train_dataloader,desc="Epoch " + str(epoch) + ":")):
            food_classification_model.zero_grad()
           
            image_batch_tensor,label_batch_tensor = data
            image_batch_tensor = image_batch_tensor.to(device)
            label_batch_tensor = label_batch_tensor.to(device)
            output = food_classification_model(image_batch_tensor)
            loss = criterion(output,label_batch_tensor)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if global_step % (616 * 2) == 0:
                evaluation(food_classification_model,transform,device)

            global_step += 1
            # print("Global Step:",global_step,"Loss:",loss)

    torch.save(food_classification_model.state_dict(),"./data/model.pth")
            
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    food_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=(256,256)),
            transforms.ToTensor()
        ]
    )

    train(food_transform,device)