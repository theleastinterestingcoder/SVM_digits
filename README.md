SVM_digits
==========

Recognizing handwritten digits using support vector machines

<h3> Introduction </h3>


SVM's are *classification* techinques commonly seen in:
<ul>
    <li> Image recognition (Face or not Face?) </li>
    <li> Protein Classification </li>
    <li> Hand Written Classification (this demo) </li>

</ul>

For more, checkout: http://scikit-learn.org/stable/modules/svm.html

<h3> Files in this directory </h3>

There are three files in this directory:
<ul>
    <li> *tutorial.py* - the tutorial on SVM from scikit</li>
    <li> *image_to_digit.py* which converts an image to digit with an SVM </li>
    <li> *svm_util.py* which supports image_to_digit.py</li>
</ul>


<h3> Installing Sci-kit </h3>

If you haven't installed sci-kit already, checkout http://scikit-learn.org/stable/install.html . The instructions there are pretty easy to follow. 


Next, install skimage. It's a useful open-source image processing library that is well integrated into scikit. We'll need it for the script that I've written. (Download Link here: http://scikit-image.org/download.html)

<h3> Running the Example Script </h3>

Once you have sci-kit installed, the example script written by Gael Varoquaux runs right out of the box:

```
python tutorial.py
```

Once you've run the script, and look at the source code and checkout the following questions. (5-10 minutes)
<ol>
    <li> How big the entire dataset? </li>
    <li> What are the dimensions of each image? </li>
    <li> How big is the training set? The testing set? </li>
    <li> Take a look at the confusion matrix generated by the script. What digits are most often confused with each other?
</ol>


<h3> Classifying your own image </h3>

To explore this SVM a little bit more, let's make some of our own data and interpret the results. Here's an example of an image I've created through photoshop. 

![alt tag](https://github.com/theleastinterestingcoder/SVM_digits/blob/master/resources/5_readme.png)

```
python image_to_digit.py 5.png
```

When creating your own image, it is _very_ important to create an image that is  _8-pixel_ by _8-pixel_ in size. Using any other image size will cause the script to crash (since the SVM has only been trained on 8x8 pizel images and doesn't know how to handle images of higher dimensions). In photoshop, this can be adjusted in Top Menu -> Image -> Canvas Size.

If you want to execute the script with your own image, use the syntax:
```
python image_to_digit.py my_file.png
```

Some interesting questions for you to try out: 
<ol>
    <li> Choose and draw your favorite number. Did the classifier guess correctly? </li>
    <li> With the same number, try drawing different variations of it. (For example, zero can be written as "0" or as "θ" </li>
    <li> Draw anything that's not a number. What did your classifier spit out? </li>
    <li> Try drawing a simple number (like the letter one), but don't do it perfectly. Try drawing it off-centered, shorter than usual, thicker than usual, slanted and etc. Do you get the number "1" from your classifier all the time?</li>
</ol>


<h3> Questions to think about: </h3>

<ul>
    <li> How well does the classifier handle "bad" inputs? </li>
    <li> What do you think the behavoir of this classifier would be if it had a poor training set? </li>
    <li> What are some issues that this classifier has? What are some ways that it intelligent and not intelligent? </li>
    <li> What's more important? Good data or good classifier? </li>
</ul>


<h3> Summary of SVM's </h3>

Advantages of SVM's:
<ul>
    <li> Effective in high dimensional spaces. </li>
    <li> Still effective in cases where number of dimensions is greater than the number of samples. </li>
    <li> Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient. </li>
    <li> Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels. </li>
</ul>


Disadvantages of SVM's:
<ul>
    <li> If the number of features is much greater than the number of samples, the method is likely to give poor 
performances. </li>
    <li> SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below). </li>
</ul>


Because SVM's work well in classification problems with a high number of dimensions, they are mostly often seen for classifying images. 


Source: http://scikit-learn.org/stable/modules/svm.html











