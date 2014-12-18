SVM_digits
==========

Recognizing handwritten digits using support vector machines

<h3> Installing Sci-kit </h3>

If you haven't installed sci-kit already, checkout http://scikit-learn.org/stable/install.html

The instructions there are pretty easy to follow. 

<h3> Running the example script </h3>

Once you have sci-kit installed, the example script written by Gael Varoquaux runs right out of the box:

'''
python plot_digits_classification.py
'''

Once you've run the script, and look at the source code (digit_classification.py) and checkout the following questions. (5-10 minutes)
<ol>
    <li> How big the entire dataset? </li>
    <li> What are the dimensions of each image? </li>
    <li> How big is the training set? The testing set? </li>
    <li> Take a look at the confusion matrix generated by the script. What digits are most often confused with eachother?
</ol>


<h3> Classifying your own image <h3>

To explore this SVM a little bit more, let's make some of our own data and interpret the results. 

First, install skimage. It's a useful open-source image processing library that is well integrated into scikit. We'll need it for the script that I've written. (Downlload Link here: http://scikit-image.org/download.html)

Use photoshop, paint, gimp (or any other image editing software), and answer the following questions:
<ol>
    <li> Choose and draw your favorite number. Did the classifier guess correctly? </li>
    <li> With the same number, try drawing different variations of it. (For example, zero can be written as "0" or as "θ" </li>
    <li> Draw anything that's not a number. What did your classifier spit out? </li>
    <li> Try drawing a simple number (like the letter one), but don't do it perfectly. Try drawing it off-centered, shorter than usual, thicker than usual, slanted and etc. Do you get the number "1" from your classifier all the time?</li>
</ol>


<h3> Questions to think about: </h3>

<ul>
    <li> How well does the classifier handle "bad" inputs? </li>
    <li> What do you think the behavoir of this classifier would be if it had a "poor" training set? </li>
    <li> What are some issues that this classifier has? </li>
    <li> What's more important? Good data or good classifier? </li>
</ul>












