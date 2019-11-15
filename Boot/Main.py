import numpy as np
import pandas as ps;

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
from blaze.expr.expressions import label

def main():
    #Load
    data = ps.read_csv("product_images.csv");
    #Separate Labels
    boots = data.loc[data['label'] == 1];
    sneakers = data.loc[data['label'] == 0];
    #Samples Count
    print("Sneakers Samples: " + str(len(sneakers)))
    print("Boots Samples: " + str(len(boots)))
    #Display 
    #display(list(sneakers.values[0]), "sneaker");
    #display(list(boots.values[0]), "boot");
    
    target = data["label"].values;

    k = 5;
    
    result = [];
    
    print("Perceptron: ");
    result.append(evaluate(Perceptron(), data.values, target, k));
    
    #trainX = preprocessing.scale(trainX)
    #testX = preprocessing.scale(testX)
    #data = StandardScaler().fit_transform(data.values)
    #data = scaler.fit_transfrom(data)
    
    print("SVC Linear: ")
    result.append(evaluate(SVC(kernel="linear", max_iter=2000), data.values, target, k));
    
    print("SVC Radial: ")
    result.append(evaluate(SVC(kernel="rbf", max_iter=2000 ), data.values, target, k));
    
    #Conclusion
    summary(result, ["Perceptron", "SVC Linear", "SVC Radial"]);

def summary(result, models):

    best_train = [float('inf'), 0]
    best_pred = [float('inf'), 0]
    best_score = [0, 0]
    
    for x in range(len(result)):
        if(result[x][0] < best_train[0]):
            best_train[0] = result[x][0]
            best_train[1] = models[x];
        if(result[x][1] < best_pred[0]):
            best_pred[0] = result[x][1]
            best_pred[1] = models[x]
        if(result[x][2] > best_score[0]):
            best_score[0] = result[x][2]
            best_score[1] = models[x]
    
    print("Best train time: " + str(best_train[0]) + ", Model: " + str(best_train[1]))        
    print("Best predict time: " + str(best_pred[0]) + ", Model: " + str(best_pred[1]))  
    print("Best score: " + str(best_score[0]) + ", Model: " + str(best_score[1]))  
    
def evaluate(model, x, y, k):
    
    fit_time = [];
    score_time = [];
    score = [];
    confusion_matrix = [];
    
    k_fold = KFold(n_splits = k);
    for x_index, y_index in k_fold.split(x):
        trainX, testX, trainY, testY = x[x_index], x[y_index], y[x_index], y[y_index];
        
        start = time.perf_counter();
        model.fit(trainX, trainY);
        end = time.perf_counter();
        
        fit_time.append(end - start);
        
        start = time.perf_counter();
        s = model.score(testX, testY)
        end = time.perf_counter();
        
        score_time.append(end - start);
        score.append(s);
        
        confusion_matrix.append(parse_confusion_matrix(model, trainX, trainY));
    
    return min_max_avrg(fit_time, score_time, score, confusion_matrix);

def parse_confusion_matrix(model, trainX, trainY):
    predicts = model.predict(trainX);
    tn, fp, fn, tp = confusion_matrix(trainY, predicts).ravel();
    return [tn, fp, fn, tp];

def display(pixelArray, title):
    del pixelArray[:1]
    plt.imshow(np.reshape(pixelArray, (28, 28)).reshape(28, 28), cmap='gray', vmin=0, vmax=255)
    plt.show()
'''
def print_confusion_matrix(confusion_matrix):
    print("True Negative: " + str(confusion_matrix[0]));
    print("False Positive: " + str(confusion_matrix[1]));
    print("False Negative: " + str(confusion_matrix[2]));
    print("True Positive: " + str(confusion_matrix[3]));
    print("\r\n")
'''   
def min_max_avrg(fit_time, score_time, score, confusion_matrix):
    
    train_time_avr = sum(fit_time)/len(fit_time);
    prediction_time_avr = sum(score_time)/len(score_time);
    score_avr = sum(score)/len(score);
    
    print("Train time array: " + str(fit_time));
    print("Min train time: " + str(max(fit_time)))
    print("Max train time: " + str(min(fit_time)))
    print("Average train time: " + str(train_time_avr));
    print("\r\n")
    
    print("Prediction time array: " + str(score_time));
    print("Min prediction time: " + str(max(score_time)))
    print("Max prediction time: " + str(min(score_time)))
    print("Average prediction time: " + str(prediction_time_avr));
    print("\r\n")
    
    print("Score array: " + str(score));
    print("Min score: " + str(max(score)))
    print("Max score: " + str(min(score)))
    print("Average score: " + str(score_avr));
    print("\r\n")
    
    print("Confusion matrix: True Negative, False Positive, False Negative, True Positive")
    print("Confusion matrix: " + str(confusion_matrix))
    print("\r\n")
    
    return train_time_avr, prediction_time_avr, score_avr;


main();