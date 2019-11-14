import numpy as np
import pandas as ps;

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import time

def main():
    #Load
    data = ps.read_csv("product_images.csv");
    
    #Separate Labels
    boots = data.loc[data['label'] == 1];
    sneakers = data.loc[data['label'] == 0];
    
    sneaker = list(sneakers.values[0]);
    boot = list(boots.values[0]);
    
    sneakers_targer = data["label"].values;
    print(sneakers_targer)
    #Samples Count
    print("Sneakers Samples: " + str(len(sneakers)))
    print("Boots Samples: " + str(len(boots)))
    
    #Display Images
    #display(sneaker, "sneaker");
    #display(boot, "boot");

    
    k = 5;
    
    #trainX, testX, trainY, testY = train_test_split(x, y, test_size=1/k);
    
    fit_time, score_time, score = eval(Perceptron(), data, sneakers_targer, k);
    print(eval(Perceptron(), data, sneakers_targer, k));
    print(score_time);
    print(score);
    #trainX, testX, trainY, testY = train_test_split(data, sneakers_targer, test_size=1/5);
    #train, test = train_test_split(data, test_size = 0.3);
    
    
    #print(len(trainX));
    #print(len(trainY));
    #print(len(testX));
    #print(len(testY))
    
    #scoring = ['precision_macro', 'recall_macro'];
    #score = cross_validate(Perceptron(), data, sneakers_targer, cv = k)
    #print(score);
    #print (score.mean());
    #print(cross_val_score(SVC(), data, sneakers_targer, cv = k));
    
    #pr = Perceptron();
    #pr.fit(trainX, trainY);
    #print(pr.score(testX, testY));
    
    #lr = LogisticRegression()
    #lr.fit(trainX, trainY);
    #print(lr.score(testX, testY));
    
def eval(model, x, y, k):
    
    fit_time = [];
    score_time = [];
    score = [];
    
    k_fold = KFold(n_splits = k);
    for x_index, y_index in k_fold.split(x):
        trainX, testX, trainY, testY = x.values[x_index], x.values[y_index], y[x_index], y[y_index];
        
        start = time.perf_counter();
        model.fit(trainX, trainY);
        end = time.perf_counter();
        
        fit_time.append(end - start);
        
        start = time.perf_counter();
        s = model.score(testX, testY)
        end = time.perf_counter();
        
        score_time.append(end - start);
        
        score.append(s);
    
    return fit_time, score_time, score;

def display(pixelArray, title):
    del pixelArray[:1]
    #im = Image.new('L', (28, 28))
    #im.putdata(tuple(pixelArray));
    #arr = np.asarray()
    plt.imshow(np.reshape(pixelArray, (28, 28)).reshape(28, 28), cmap='gray', vmin=0, vmax=255)
    plt.show()
    #im.show()
   
   
'''
def magni(image, width, height, incrX, incrY):
    newImage = [];

    for y in range(0, height):
        for x in range(0, width):
            for yIncr in range(0, incrY):
                for xIncr in range(0, incrX):
                    pixel = image[x + y * width];
                    pixIndex = (x * incrX + xIncr) + ((y * incrY + yIncr) * width);
                    newImage.insert(pixIndex, pixel);
    return newImage;
'''
    
main();