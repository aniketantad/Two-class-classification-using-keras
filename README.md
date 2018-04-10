In this code I have created CNN using Keras for two class image classification model.

Required packages: Numpy, Keras, comet_ml(Optional)

For this code to run you need to create a following folder structure in the current code directory:

Test
    Class1
          image1
          image2
          .
          .
          .
          imageN
    Class2
          image1
          image2
          .
          .
          .
          imageN
 Training 
     Class1
          image1
          image2
          .
          .
          .
          imageM
    Class2
          image1
          image2
          .
          .
          .
          imageM
 
then you need to make changes in following two lines:
nb_train_samples = 20000(Class1 + Class2) training images
nb_test_samples = 2000(Class1 + Class2) testing images

If comet_ml package is included in the code then create account on https://www.comet.ml/ and then add the experiment line mentioned for your account.

CometML helps with different hyperparameter values and help us to visualize the network.
