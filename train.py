import argparse
from pyexpat import model
from statistics import mode
from cv2 import log
import numpy as np
from sklearn import model_selection
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from loss import Front_loss, GA_loss, GA_val_loss
from model import get_front_model, get_gender_age_model  
from Dataset import GADataset, FrontDataset
import time, os, cv2
from visualization import *



def train(model_type):
    if model_type == 'GA':
        dataset  = GADataset(img_size)
        net = get_gender_age_model()
        criterion = GA_loss
    elif model_type == 'Front':
        dataset  = FrontDataset(img_size)
        net = get_front_model()
        criterion = Front_loss
    
    print('dataset size: ', len(dataset))
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    print('train size: ', train_size)
    print('val size: ', val_size)
    train_dataset, val_dataset = random_split(dataset,[train_size,val_size])
    test_dataset = val_dataset

    #check gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device : ', device)

    #set hyperparameter
    EPOCH = 30
    pre_epoch = 0
    BATCH_SIZE = 64
    LR = 0.01

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    #labels in CIFAR10
    #classes = ('front','side','back','hat','unknown')
    #classes = ('student','young','middle','old')
    #define ResNet18   

    #define loss funtion & optimizer
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    
    #train
    best_epoch = 0
    best_accuracy = 0.0
    list_epoch, list_train_loss, list_gender_acc, list_age_acc, list_front_acc = [], [], [], [], []

    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        sum_age_loss = 0.0
        sum_gender_loss = 0.0
        sum_front_loss = 0.0
        correct = 0.0
        total = 0.0
        last_train_count = 0
        last_val_count = 0
        for i, data in enumerate(trainloader, 0):
            #prepare dataset
            if model_type == 'GA':
                inputs, gender, age = data
                inputs, gender, age = inputs.to(device), gender.to(device), age.to(device)
            elif model_type == 'Front':
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            #forward & backward
            outputs = net(inputs.to(device))
            if model_type == 'GA':
                targets = (gender.to(device),age.to(device))
            elif model_type == 'Front':
                targets = labels
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            #print ac & loss in each batch
            sum_loss += loss.item()
            if model_type == 'GA':
                _, gender_predicted = torch.max(outputs[0].data, 1)
                _, age_predicted = torch.max(outputs[1].data, 1)
                total += gender.size(0)
            elif model_type == 'Front':
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
            #print('[epoch: {}, iter: {}] Loss: {} '.format(epoch+1, (i+1+epoch*length), sum_loss/(i+1))) 
            last_train_count = i
            
        #get the ac with testdataset in each epoch
        print('Waiting Test...')
        with torch.no_grad():
            if model_type == 'GA':
                pth_name = '/ga_' + str(img_size) + '.pth'
            elif model_type == 'Front':
                pth_name = '/front_' + str(img_size) + '.pth'
            gender_acc = 0
            age_acc = 0
            front_acc = 0
            count = 0
            for data in valloader:
                net.eval()
                if model_type == 'GA':
                    inputs, gender, age = data
                    inputs, gender, age = inputs.to(device), gender.to(device), age.to(device)
                    outputs = net(inputs.to(device))
                    targets = (gender.to(device), age.to(device))
                    gender_loss, age_loss = GA_val_loss(outputs, targets)
                    sum_gender_loss += gender_loss.item()
                    sum_age_loss += age_loss.item()
                    _, gender_predicted = torch.max(outputs[0].data, 1)
                    _, age_predicted = torch.max(outputs[1].data, 1)
                    if gender_predicted == gender :
                        gender_acc += 1
                    if age_predicted == age :
                        age_acc += 1
                elif model_type == 'Front':
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    if predicted == labels:
                        front_acc += 1
                
                #print('gender_predicted : {} gender : {}'.format(gender_predicted, gender))
                count +=1

            list_epoch.append(epoch)
            list_train_loss.append(sum_loss / (last_train_count))
            if model_type == 'GA':
                list_gender_acc.append((sum_gender_loss) / count)
                list_age_acc.append((sum_age_loss) / count)
                #print('list_train_loss',list_train_loss, 'list_gender_acc',list_gender_acc, 'list_age_acc',list_age_acc)
                if best_accuracy < (100 * age_acc / count ) :
                    best_accuracy = (100 * age_acc / count )
                    torch.save(net.state_dict(), './models/' + str(model_type) + pth_name)
                    print('GA weights saved!')
                    print('saved at: /models/GA' + pth_name)
                print('Test\'s ac is: gender {} age {}'.format((100*gender_acc/count),(100*age_acc/count)))
                # draw graph!
                if (epoch+1) % EPOCH == 0 :
                    make_graph(list_epoch, list_train_loss, list_gender_acc, list_age_acc, 'train loss', 'gender validation loss', 'age validation loss')
            elif model_type == 'Front':
                list_front_acc.append((front_acc))
                if best_accuracy < (100 * front_acc / count ) :
                    best_accuracy = (100 * front_acc / count )
                    torch.save(net.state_dict(), './models/' + str(model_type) + pth_name)
                    print('Front weights saved!')
                    print('saved at: /models/Front' + pth_name)
                print('Test\'s ac is: front {}'.format((100*front_acc/count)))
                if (epoch+1) % EPOCH == 0 :
                    make_graph(list_epoch, list_train_loss, list_front_acc, list_front_acc, 'train loss', 'front validation loss', 'front validation loss')

    #torch.save(net.)
    print('Train has finished, total epoch is %d' % EPOCH)
    #torch.save(net.state_dict(),'./checkpoint_new.pth')

    return pth_name

def test(model_type, saved_model):
    saved_model = os.path.join('./models/' + model_type + saved_model)
    if model_type == 'GA':
        test_dataset = GADataset(img_size)
    elif model_type == 'Front':
        test_dataset = FrontDataset(img_size)
    
    #check gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    testloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)
    count = 0
    a_count = 0
    f_count = 0

    #labels in CIFAR10
    if model_type == 'GA':
        net = get_gender_age_model()
    elif model_type == 'Front':
        net = get_front_model()
    net.load_state_dict(torch.load(saved_model))

    with torch.no_grad():
        correct = 0
        total = 0
        g_correct = 0
        g_total = 0
        a_correct = 0
        a_total = 0
        for data in testloader:
            net.eval()
            if model_type == 'GA':
                images, gender, age = data
                images, gender, age = images.to(device), gender.to(device), age.to(device)
            elif model_type == 'Front':
                images, labels = data
                images, labels = images.to(device), labels.to(device)
            #print(images.shape)
            tf = transforms.ToPILImage()
            tf_image = tf(images[0])
            #tf_image.show(title=label_name)
            
            #time.sleep(2)

            start_time = time.time()
            outputs = net(images)
            print("Testing time : ",time.time() - start_time)

            if model_type == 'GA':
                _, gender_predicted = torch.max(outputs[0].data, 1)
                _, age_predicted = torch.max(outputs[1].data, 1)
                if ( gender[0].item() == 0 ) :
                    label_gt = "male" 
                else : 
                    label_gt = "female"

                if ( gender_predicted == 0 ):
                    predict =  "male" 
                else : 
                    predict = "female"


                if predict != label_gt :
                    print("gt : {}, predict : {} wrong".format(label_gt, predict))
                    #tf_image.show()
                    #time.sleep(1)
                    # print(type(images), type(images[0]))
                    # print(images.size())
                    # print(type(np.array(tf_image)))
                    # print(np.array(tf_image).shape)
                    cv2.imwrite('./dataset/predicted_failed_img/gender/gender_' + str(count) + '_label_' + label_gt + '_predict_' + predict + '.png', cv2.cvtColor(np.array(tf_image), cv2.COLOR_BGR2RGB)) 
                    count += 1

                if ( age[0].item() == 0 ) :
                    label_a = "student" 
                elif ( age[0].item() == 1 ):
                    label_a = "young"
                elif ( age[0].item() == 2 ):
                    label_a = "middle"                    
                else : 
                    label_a = "old"

                if ( age_predicted == 0 ):
                    predict_a =  "student" 
                elif ( age_predicted == 1):
                    predict_a = "young"
                elif ( age_predicted == 2):
                    predict_a = "middle"
                else : 
                    predict_a = "old"

                if predict_a == label_a :
                    print("age : {}, predict : {} correct".format(label_a, predict_a))
                else :
                    print("age : {}, predict : {} wrong".format(label_a, predict_a))
                    #tf_image.show()
                    #time.sleep(1)
                    cv2.imwrite('./dataset/predicted_failed_img/age/age_' + str(a_count) + '_label_' + label_a + '_predict_' + predict_a + '.png', cv2.cvtColor(np.array(tf_image), cv2.COLOR_BGR2RGB)) 
                    a_count += 1
                
                g_total += gender.size(0)
                g_correct += (gender_predicted == gender).sum()
                a_total += age.size(0)
                a_correct += (age_predicted == age).sum()
                
            elif model_type == 'Front':
                _, predicted = torch.max(outputs.data, 1)
                if ( labels[0].item() == 0 ) :
                    label_gt = "front" 
                else : 
                    label_gt = "not_front"

                if ( predicted == 0 ):
                    predict =  "front" 
                else : 
                    predict = "not_front"
                    
                if predict != label_gt :
                    print("gt : {}, predict : {} wrong".format(label_gt, predict))
                    #tf_image.show()
                    #time.sleep(3)
                    cv2.imwrite('./dataset/predicted_failed_img/front/front_' + str(f_count) + '_label_' + label_gt + '_predict_' + predict + '.png', cv2.cvtColor(np.array(tf_image), cv2.COLOR_BGR2RGB)) 
                    f_count += 1
                total += labels.size(0)
                correct += (predicted == labels).sum()
        if model_type == 'GA':
            print('Total size: ', g_total,'right: ', g_correct,'wrong: ', count)
            print('Test\'s gender is: %.3f%%' % (100 * g_correct / g_total))
            print('Total size: ', a_total,'right: ', a_correct,'wrong: ', a_count)
            print('Test\'s ac is: %.3f%%' % (100 * a_correct / a_total))
        elif model_type == 'Front':
            print('Total size: ', total,'right: ', correct,'wrong: ', f_count)
            print('Test\'s ac is: %.3f%%' % (100 * correct / total))


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='train With Pytorch')
    parser.add_argument("--model_type", default="GA", type=str,
                        help='Specify model type.')
    parser.add_argument("--image_size", default=64, type=int,
                        help='Specify image size.')

    args = parser.parse_args()
    img_size = args.image_size
    model_type = args.model_type

    if not model_type == "GA" and not model_type == "Front":
        print('올바른 모델 타입을 입력하세요.')
        raise

    if not img_size % 2 == 0:
        print("image size 는 짝수로 맞춰주세요.")
        raise

    print('model_type: ', model_type)
    print('image_size: ', img_size)
    saved_model = train(model_type)
    test(model_type, saved_model)