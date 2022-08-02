import torch
import torch.nn as nn
import argparse
import numpy as np

def GA_loss(outputs, targets):
    gender_output, age_output = outputs
    gender_gt, age_gt = targets
    gender_loss = nn.CrossEntropyLoss()(gender_output,gender_gt)
    age_loss = nn.CrossEntropyLoss()(age_output,age_gt)
    #print('gender loss : {}, age loss : {}'.format(gender_loss, age_loss))
    return (gender_loss + age_loss )/2

def GA_val_loss(outputs, targets):
    gender_output, age_output = outputs
    gender_gt, age_gt = targets
    gender_loss = nn.CrossEntropyLoss()(gender_output,gender_gt)
    age_loss = nn.CrossEntropyLoss()(age_output,age_gt)
    #print('gender loss : {}, age loss : {}'.format(gender_loss, age_loss))
    return gender_loss, age_loss


def Front_loss(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)


def cross_entropy(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

def mean_squared(outputs, targets):
    return nn.MSELoss()(outputs, targets)


if __name__ == '__main__':
    print('loss.py') 
    parser = argparse.ArgumentParser(description="calculate loss")
    parser.add_argument("--loss_function", default="Cross Entropy", type=str,
                        help='Specify loss function.(Cross Entropy / Mean Squared / Binanry Cross Entropy)')
    args = parser.parse_args()

    loss_function = args.loss_function
    output = torch.Tensor([
            [0.8982, 0.805, 0.6393, 0.9983, 0.5731, 0.0469, 0.556, 0.1476, 0.8404, 0.5544],
            [0.9457, 0.0195, 0.9846, 0.3231, 0.1605, 0.3143, 0.9508, 0.2762, 0.7276, 0.4332]])
    target = torch.LongTensor([1, 5])
    if loss_function == "Cross Entropy":
        print(loss_function+'\n')
        print("output: ", output, "\ntarget: ", target, '\n')
        print(cross_entropy(output, target))
    elif loss_function == "Mean Squared":
        print(loss_function+'\n')
        inputs = torch.randn(2, 4, requires_grad=True)
        targets = torch.randn(2, 4)
        outputs = mean_squared(inputs, targets)
        outputs.backward()

        print('input: ', inputs)
        print('target: ', targets)
        print('output: ', outputs)
    elif loss_function == "Binary Cross Entropy":
        print(loss_function)
    # 로스값 출력
    print()
    