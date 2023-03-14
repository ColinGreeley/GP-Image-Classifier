
from image_classifier import Classifier
import tensorflow_datasets as tfds
import numpy as np


def get_imagenet_data():
    ds, info = tfds.load('imagenet2012', with_info=True)
    X = np.array([feature['image'] for feature in ds], dtype=np.uint8)
    Y = np.array([feature['label'] for feature in ds], dtype=np.float32)
    return (X, Y)
    
    

if __name__ == "__main__":
    
    images, labels = get_imagenet_data()
    classifier = Classifier(image_rez=(224,224), batch_size=175, ensemble_size=7)
    classifier.name = "ImageNet_Classifier"
    classifier.test(images, labels, class_labels=set(labels))