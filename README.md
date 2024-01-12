# Calcification_Classification
This folder contains all the code require for a few different deep learning models that will predict the level of calcification based sole on the non-gated CT scans. First, run the segmentator.py code on your images. This will crop your images so that they only contain the heart and surrounding vessels. Then, upload your cropped images to the scrath directory. 

Once your images have been processed this way, you're ready to run the model. The default is a ResNet, although you can easily switch to a DenseNet. After some brief data augmentation involving rotating and stretching the images, the model will train on the portion earmarked for training. 
