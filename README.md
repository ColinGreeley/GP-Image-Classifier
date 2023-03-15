# GP-Image-Classifier
General Purpose Image Classification Library: Using only an image dataset as input, this model will automatically build and train a CNN using the latest best methods while also being optimized for the resolution of the training data.

example.py
```python

from image_classifier import Classifier
import tensorflow_datasets as tfds
import numpy as np


def get_imagenet_data():
    ds, info = tfds.load('imagenet2012', with_info=True)
    X = np.array([feature['image'] for feature in ds], dtype=np.uint8)      # X shape: (1280000, 244, 244, 3)
    Y = np.array([feature['label'] for feature in ds], dtype=np.float32)    # Y shape: (1280000, 1000)
    return (X, Y)
    
    

if __name__ == "__main__":
    
    images, labels = get_imagenet_data()
    classifier = Classifier(image_rez=(224,224), batch_size=175, ensemble_size=7)
    classifier.name = "ImageNet_Classifier"
    classifier.train(images, labels, class_labels=set(labels))
    classifier.test(images, labels, class_labels=set(labels))
```

In this section of python code, the images and labels for ImageNet is loaded in and a new classifier is built for this data. 

The train function completes the entire training pileline by structuring training data generators using bagging for ensemble validation and majority voting. When the average ensemble F1-score has converged, the trainable parameters are optimal and the model is saved.

The test function is a variant of the train function where the input data is randomly split 80/20 to train and test sets. Evaluation on the test set is recored k times and averaged using random sampling to create the train/test slit each k times.

Test() can be run independant of train() if model evaluation is all that is desired.


## Requirements
* Label data must be one-hot encoded since multi-label classification is supported.
* batch_size should be divisible by ensemble size
* image pre-processing should not occur. Model expects channel range [0,255], therefore uint8 datatype can be used to save memory
