# Image Classification with Bag of Visual Words

This project utilizes the Bag of Visual Words approach to classify images. In addition to the basic approach, I added in an implementation of spatial pyramid matching to attempt to get a sense of spatial representation of the image features.

In addition to the files with all the code for the project, I walk you throgh an example of using Bag of Visual Words to classify insects from the ImageNet dataset. ***(Coming Soon!)***

**Packages Used**
- OpenCV
- Scikit Learn (MiniBatchKMeans, SVC, Random Forest)
- Numpy
- Pandas

## A Bag of Visual Words Primer

So what exactly is this mystical Bag I speak of?  Essentially, it is a collection of similar features found in the images of your dataset. The true mystique, however, is not in the Bag itself; rather, the mystique is in the creation of this Bag and its contents.

And that is precisely what I will describe next!

### Step 1: Create a Set of Descriptor Vectors for Each Image

This step involves using alogorithms such as SIFT and/or MSER to locate the main features of the image, known as keypoints. This will amount to finding areas like corners and blobs that are most fundamental to distinguishing one object from another. For this, we can utilize OpenCV's built-in implementations of such algorithms, with a few minor tweaks.

OpenCV also enables us to create descriptor vectors for each of these keypoints.  These 128x1 vectors "describe" the keypoint by giving intensity characterization in the form of an 8-bin histogram for each square in a 4x4 grid around the keypoint.

For each image, we may end up with dozens, or even hundreds, of these 128x1 vectors, which we will utilize in Step 2.

### Step 2: Create the Feature Space for All Images

In this step, we throw all of our descriptor vectors for EACH image into a "bag" and perform KMeans (I used scikit-learn's MiniBatchKMeans) clustering to pull out k groups of related features.  The resulting clusters will be our set of features, which we will use in step 3.

To help gain intuitive understanding, you can imagine that one cluster might represent a cat's ear or the tip of a dog's tail.

### Step 3: Build a Histogram of Features for Each Image

For this step, we will return to each individual image and match every descriptor vector in a given image to the feature cluster with the greatest similarity. This will give us a k-bin histogram for each image, representing the number of each feature contained in the image.

This feature vector will then be fed into our classifier in the fourth and final step.

### Step 4: Fit a Classifier

We now have a feature vector for each image and are ready to fit a classifier.  My best results came from a Support Vector Classifier, using a radial kernel, however further experimentation may lead to some improvements.  With the classifier in place, we can finally predict the class of an image.

## Additional Comments

Admittedly, Bag of Visual Words is no longer the best practice in image classification.  The ImageNet competition has fostered incredibly rapid growth in the field, leading to many improvements in the basic approach outlined above.  Starting with AlexNet in 2012, deep convolutional networks have dominated the world of image classification.

That said, a lot can be learned about computer vision by undertaking this type of project without the use of deep ConvNets.  One can really gain an appreciation for the kinds of features kernels may detect, as well as an deeper understanding of some of the challenges in building learning algorithms to aid a computer in making sense of the visual world.


