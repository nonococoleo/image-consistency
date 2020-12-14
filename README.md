# Image Consistency

## Environment
- Python 3.8

## Models
### Feature Model
The Feature Model encodes image patches into a feature vector for further steps.
#### Comparison Model
The Comparison Model takes two patches and output a feature vector indicates their similarity.
#### Prediction Model
The Prediction Model takes one patch and outputs a one-dimensional vector. A feature vector between two patches can be calculated by comparing the corresponding output vectors.
### Consistency Model
The Consistency Model depends on the feature vector and gives a Consistency score of two patches.

## DataSet
### Image
Download more training images for Feature Model by running download_image script.
``` shell
python3 download_image.py
```
### EXIF
Extract EXIF information from images and perform uniform distribution by running extract_exif script.
``` shell
python3 extract_exif.py
```
## Training
### Feature Model
Run the train_comparsion or train_prediction script to train the corresponding model.
``` shell
python3 train_comparison.py
python3 train_prediction.py
```
### Consistency Model
Run the train_consistency script to train the Consistency Model.
``` shell
python3 train_consistency.py
```

## Evaluating
Run the evaluation script to get the heatmap about the overall consistency scores of different areas in the given image.
``` shell
python3 test_consistency.py
```
