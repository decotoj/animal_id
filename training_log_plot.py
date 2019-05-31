import os 
from collections import defaultdict   
import matplotlib.pyplot as plt
import pylab 


MODEL_DIR = 'models/'

valLoss = defaultdict(list)
valAcc = defaultdict(list) 
trnLoss = defaultdict(list)
trnAcc = defaultdict(list)
keys = []

#Get List of All Image Files in Raw Data Directory
files = []
for root, dirs, file in os.walk(MODEL_DIR):
    for i in range(len(file)):
        if '.txt' in file[i]:
            with open(MODEL_DIR + file[i], 'r') as f:   
                ln = f.readlines()

            keys.append(file[i])
            for q in range(len(ln)):
                if '########' in ln[q] and 'ln.append' not in ln[q]:
                    ln = ln[q+1:]
                    break
            for n in ln:
                if 'train' in n:
                    trnLoss[keys[-1]].append(float(n.split(': ')[1].split(' ')[0]))
                    trnAcc[keys[-1]].append(float(n.split(': ')[2].split(' ')[0]))
                elif 'val' in n:
                    valLoss[keys[-1]].append(float(n.split(': ')[1].split(' ')[0]))
                    valAcc[keys[-1]].append(float(n.split(': ')[2].split(' ')[0]))

#Print Validation Accuracy
for k in keys:
    try:
        print(k.replace('model_','').split('.')[0],' : Val Acc: ', max(valAcc[k]))
    except:
        pass

c=['r','g','b','k','y','m', 'c']

i = 0
for k in keys:
    x = list(range(0,len(trnLoss[k])))
    pylab.plot(x, trnLoss[k], linestyle='-', marker='o', color=c[i], label=k.replace('model_','').split('.')[0])
    pylab.plot(x, valLoss[k], linestyle='--', marker='v', color=c[i])
    i+=1
pylab.legend(loc='upper right')
pylab.xlabel('Epoch')
pylab.ylabel('Loss')
pylab.title('Loss: Training -> Circles, Validation -> Triangles')
pylab.grid()
pylab.show()

i = 0
for k in keys:
    x = list(range(0,len(trnLoss[k])))
    pylab.plot(x, trnAcc[k], linestyle='-', marker='o', color=c[i], label=k.replace('model_','').split('.')[0])
    pylab.plot(x, valAcc[k], linestyle='--', marker='o', color=c[i])
    i+=1
pylab.legend(loc='lower right')
pylab.xlabel('Epoch')
pylab.ylabel('Accuracy')
pylab.title('Accuracy: Training = Solid Line, Validation = Dashed Line')
pylab.grid()
pylab.show()