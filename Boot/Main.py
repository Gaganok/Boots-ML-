import numpy as np
import scipy.misc as smp
import pandas as ps;
from PIL import Image
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits

def main():
    #Load
    data = ps.read_csv("product_images.csv");
    
    digits = load_digits(True);
    #print(digits.data)
    print(digits.target)
    
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
    
    trainX, testX, trainY, testY = train_test_split(data, sneakers_targer, test_size=0.3);
    #train, test = train_test_split(data, test_size = 0.3);
    
    
    print(len(trainX));
    print(len(trainY));
    print(len(testX));
    print(len(testY))
    
    #print(cross_val_score(Perceptron(), train, test));
    pr = Perceptron();
    pr.fit(trainX, trainY);
    print(pr.score(testX, testY));
    
    #lr = LogisticRegression()
    #lr.fit(trainX, trainY);
    #print(lr.score(testX, testY));
    
    
def display(pixelArray, title):
    del pixelArray[:1]
    im = Image.new('L', (28, 28))
    im.putdata(tuple(pixelArray));
    im.show()
   
   
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