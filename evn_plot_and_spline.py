import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import operator

#outdir1 = '/home/joaobiu/pic/results/evn_final_1/'

#outdir1 = 'paper_evn_sweeping_1/'

import sys
if len(sys.argv) > 1:
    outdir1 = sys.argv[1]
else:
    print("Please provide the directory of the data as an argument when running the script in the terminal.")
    sys.exit()
if outdir1[-1] != '/':
    outdir1 += '/'



def Read_Two_Column_File(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))

    return x, y

x1, y1 = Read_Two_Column_File(outdir1 + 'data.txt')

def minimum(x, y, savename, **kwargs):
    min_x = x[y.index(min(y))]
    min_y = min(y)
    
    plt.subplots(figsize=(6, 5))
    plt.scatter(x , y, color = "black", marker=".")
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.scatter(min_x, min_y)
    plt.axvline(x=min_x, color='r', linestyle='--')
    plt.xlabel("Rescaling Factor (RF)", fontsize=16)
    plt.ylabel("Objective Function (J)", fontsize=16)
    plt.xticks(kwargs["ticks"])
    plt.savefig(savename, dpi = 300, bbox_inches = 'tight')

    return min_x, min_y

def sort(x, y):
    x_sort = x

    y_sort = y

    L = sorted(zip(x,y), key=operator.itemgetter(1))

    new_x, new_y = zip(*L)
    data = np.column_stack([new_x, new_y])
    datafile_path = outdir1 + "sorted_values.txt"
    np.savetxt(datafile_path , data, fmt=['%f','%e'])

# if the first value is higher than the last value, or the first value is lower than the last value
# we cut the data at the first value that is lower than the last value

if y1[0] > y1[-1]:
    x2 = []
    y2 = []
    for i in range(len(y1)):
        if y1[i] < y1[-1]:
            x2.append(x1[i])
            y2.append(y1[i])
elif y1[0] < y1[-1]:
    x2 = []
    y2 = []
    for i in range(len(y1)):
        if y1[i] < y1[0]:
            x2.append(x1[i])
            y2.append(y1[i])
else:
    x2 = x1
    y2 = y1

x_min, y_min = minimum(x2, y2, savename = outdir1 + "opt_evn_factor_cut.png", ticks=np.linspace(0.145, 0.155, 6, endpoint=True))

data = np.column_stack([x_min, y_min])
datafile_path = outdir1 + "minimum.txt"
np.savetxt(datafile_path , data, fmt=['%f','%e'])

sort(x1, y1)

print(f"{x_min} {y_min}")
