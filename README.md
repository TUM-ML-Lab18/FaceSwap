# General
If you use Pycharm for developing mark the `implementation` folder as root.

# Preprocessing
Before you are able to train models, you have to preprocess your data. Therefore you have to store all the images of
your dataset in a subfolder called 'raw'. Place all images directly in the raw folder and do not use subfolders.

```
dataset/
`-- raw
    |-- image1.jpg
    |-- image2.jpg
    .
    .
    .
    `-- image9999.jpg
```

In the next step you have to define the path to your dataset. This can be done in `Configuration/config_general`:

```python
DATASET = "/path_to/dataset"
ROOT = Path(DATASET)
```

Now you need to run the script `preprocess.py` and wait until your data is processed:
```python
(face) max@MaxLaptop:~$ python preprocess.py 
 |████████████████████████████████████████████████████████████████████████████████████████████████████| 100.0%                                                                                                                                                          
All images processed. Storing results in NumPy arrays...                                                                                                                                                                                                                
Preprocessing finished! You can now start to train your models!  
```

After the preprocessing your dataset folder should contain several subfolders with images in different resolutions and
a lot of NumPy arrays with extracted features.

# Architecture
### FaceExtractor & FaceReconstructor

![FaceExtractor FaceReconstructor](images/FaceExtractor_FaceReconstructor.jpg "FaceExtractor & FaceReconstructor")

We use the depicted pipeline during preprocessing and training to produce images, that are as easy as possible to
be modeled using our neural nets. The second part of that pipeline fits generated images back into the original scene.
This module is built using classical computer vision methods. For the extraction of facial landmarks we used the great
face recognition package from [Adam Geitgey](https://github.com/ageitgey/face_recognition). We use those landmarks to determine the
position of faces in images, but also as input features for our networks.