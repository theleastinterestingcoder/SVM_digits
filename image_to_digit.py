'''
    image_to_digit.py

    written by Quan Zhou on 12/16/14

    Converts an 8x8 pixel image to numpy. Make sure 
'''

import numpy as np
from skimage import io
from svm_util import train_digit_classifier

# Name of the image we want to classify
fname = '5.png'     # IMPORTANT! This can be any image format as long as it is a 8-pixel by 8-pixel image. 

# Train the classifier
classifier = train_digit_classifier()   #Check out digit_classifier.py for more details

#Convert .png file -> python readible matrix (numpy)
pixel_data = io.imread(fname, as_grey='true')

#Do some data processing over our image
pixel_data = (1-pixel_data)                     # Invert the image
pixel_data = pixel_data/pixel_data.max()        # Normalize the data
pixel_data = pixel_data * 16                    # Rescale to 16.0 (so our svm can understand it)
pixel_data = np.round(pixel_data)               # convert decimal -> int

#Reshape the 8x8 -> 16x1 vector
pixel_vector = pixel_data.reshape(1,-1)

#Make a prediction!  (Make sure you ran digit_classification first!)
guess = classifier.predict(pixel_vector)

print "SVM predict the digit as: %s" % (guess[0])