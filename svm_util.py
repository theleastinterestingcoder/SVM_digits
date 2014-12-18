'''
    digit_classifier.py

    written by Quan Zhou

    This an adapted version of "digit_classifier.py" written Gael Varoquaux. 
    This contains a function by that will train a SVM on SKlearn's "digit dataset." 
'''

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt

    
# This function trains a classifier over SKlearn's digit dataset 
def train_digit_classifier():
    # Choose a fraction of the images to learn on (the dirty work gest done here)
    frac = 3/4.0;

    # The digits dataset
    digits = datasets.load_digits()

    # To apply a classifier on this data, we need to flatten the image into a one-dimensional vector,
    #to  turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))   #Reshape changes the shape of the matrix

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)       #Gamma is the "regularization" level that prevents "overfitting"

    # Now that everthign is set up, let's train our classifier!
    classifier.fit(data[:n_samples * frac], digits.target[:n_samples * frac]) 

    # Easy, return the classifier 
    return classifier