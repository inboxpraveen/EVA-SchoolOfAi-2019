# Assignment to do: 

1. We have considered many many points in our last 4 lectures. Some of these we have covered directly and some indirectly. They are:

   1. How many layers,
   2. MaxPooling,
   3. 1x1 Convolutions,
   4. 3x3 Convolutions,
   5. Receptive Field,
   6. SoftMax,
   7. Learning Rate,
   8. Kernels and how do we decide the number of kernels?
   9. Batch Normalization,
   10. Image Normalization,
   11. Position of MaxPooling,
   12. Concept of Transition Layers,
   13. Position of Transition Layer,
   14. Number of Epochs and when to increase them,
   15. DropOut
   16. When do we introduce DropOut, or when do we know we have some overfitting
   17. The distance of MaxPooling from Prediction,
   18. The distance of Batch Normalization from Prediction,
   19. When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
   20. How do we know our network is not going well, comparatively, very early
   21. Batch Size, and effects of batch size
   22. When to add validation checks
   23. LR schedule and concept behind it
   24. Adam vs SGD
   25. etc (you can add more if we missed it here)

2. As we have gone through the 7 code files above, we'd like you to develop an intuition on when to introduce these. We would never want to start with everything enabled (maybe for atleast 3-4 months). Considering that, we want to think about these things in order, we would like you to:

   1. Write your Assignment 3 again such that:

      1. Finally

          

         after 4 code iterations achieve:

         1. 99.4% accuracy
         2. Less than 15k Parameters
         3. Have started from a Vanilla network (no BN, DropOut, LR, larger batch size, change in Optimizer, etc)

      2. Make sure you are tracking your code's performance, and writing down your observations are you achieve better or worse results

      3. Your second code can only have max 3 improvements over first one, third can have only max 3 over second and so on. 

      4. All of your iterations are in different files and named properly like First, Second, etc

      5. All of you iterations are in a single folder

      6. All of your iterations have a Header note, describing what all you are planning to do in this code

      7. All of your code is very well documented

      8. There is a

          

         readme

          file describing:

         1. the above mentioned (at least 24) points. The title of this file is 

            Architectural Basics. 

            You need to:

            1. Put them in order in which you will think about them or execute them in your experiments
            2. Thought behind that order

3. **This is a slightly time-consuming assignment, please make sure you start early.** 

4. **200 points on your code and Documentation, and 200 points on your ReadMe File.** (Subtractions in case your readme does not have proper formatting and does not look formal/professional)

# Solution:

Let us start by making a network in which we try to aim at the given accuracy. To do so, we just need to design a simpler vanilla network which, just might, give us the desired accuracy.



Choosing number of layers? Well , it totally depends on problems that we work on. For MNIST, as i think the range for number of layers should be 3 to 6.



For designing a Vanilla network, we need an max pool, 1x1 and 3x3 convolutional layers. These layers generally help in reducing image size but also varies the number of features we are trying to extract. Generally, max pooling is applied after first few convolutional layers and not at the last layers, so, max pool is used in between layer stack. We use 1x1 for bottleneck situations. So whenever we need to convert large number of channels to less, we use 1x1.



Receptive field is what our our last layer is able to see. We calculate receptive field to judge how well a network might do and its very important because sometimes the objects are big and we do not want out receptive field to be too small to predict that object.



SoftMax is used at the last layer(prediction layer). It runs a probability like distribution, which results to seem more accurate however fake. Learning rate is something we need to see the problem, the images and also sometimes plot metrices to judge what learning rate will be suitable for a particular type of network and what learning rate for particular problem. 



kernels are decided based on the dataset. If some images are very hard (or demand more details to be extracted), we need more kernels and vice versa. Batch normalization is used after every convolution layer except the last layer because on the last layer, we do predictions, so we do not need to normalize our output values. Image Normalization is something again based on the dataset. If the dataset does not have normally distributed pixel values, we need to normalize them so as to overcome high variance.



Transition layers are used when we want to move from more channels to lesser channels (we can also move from less channels to more channels if we know what we are doing). Transition layers are used from starting block of the network, but the last layer.



Number of epochs again depend on the dataset. We might want to plot number of epochs and accuracy of the model. This requires a hit and trial method of checking number of different epoch sizes, and based on the graph, we need to decide how many epochs are good enough for the network.



Dropouts are used if we see a large training and validation gap. They are used based on users preference, but many people use it after 2 or 3 convolution layers. Also dropouts are not used around the last (prediction) layer.



To know we are not going to do well or might do well, we build a series of networks, tweak parameters such as kernels, Batch normalization, max pool and see if the initial epochs results in better accuracy compared to the previous networks. If it does, we are moving in right direction, else not and we might need redesign or tweak heavily. Batch sizes are Hardware and dataset dependent. If a large dataset have very similar or easy images, any batch size would produce, but incase the images are hard to predict, we may need little high batch size so that we can feed our network to train on various different types of images, again if hardware allows us to.



Validation checks are used after every epoch if the dataset is small and easy. But for a hard dataset, do not want to validate as it requires more time.  Learning rate scheduler is used if we already the network performance and the gradients. If the gradients are learning too slow, then we can initially learn at higher rate and then slowly decrease.



Adam and SGD, both are good. SGD tends to output slowly whereas adam is faster. So if we are working on network where accuracy might not be a very bigger issue, we prefer adam. If SGD is used, we might take more time but it will also produce very good results. So it depends but both are equally good and have their respective backdrops.