import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def make_graph(list_a, list_b, list_c, list_d, label_a, label_b, label_c):
    plt.figure('Training')
    make_plot(list_a, list_b, label_a)
    print(list_a)
    make_plot(list_a, list_c, label_b)
    make_plot(list_a, list_d, label_c)
    #plt.title('Train visualization graph')
    plt.xlabel('Epoch')
    #plt.ylabel('Cross entropy loss')
    plt.legend(loc='upper right')
    plt.show()


def make_plot(list_a, list_b, label_str):
    plt.plot(list_a, list_b, marker='o', label=label_str)

if __name__ == "__main__":  
    list_a = list(range(0, 30))
    list_b = list(range(60, 0, -2))
    list_c = list(range(90, 0, -3))
    list_d = list(range(120, 0, -4))
    make_graph(list_a, list_b, list_c, list_d, 'train loss', 'gender validation', 'age validation')
