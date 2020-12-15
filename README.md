# Detecting Sliced Images

## Environment
- Python 3.8

## Models
#### EXIF based approach:

Predicts the EXIF information of patches first. Then use this information to tell whether the image is sliced.

- ##### Feature models (extract features at the first stage)

  Feature models encode image patches into a feature vector which is then passed to the consistency model.
  ###### Comparison model

  - The Comparison model takes two patches and predicts a boolean feature vector. This features vector measure whether these two patches have the same value on each EXIF field.

  ###### Prediction model

  - The Prediction model takes one patch and predicts the value for each EXIF field.

- ##### Consistency model (derive consistency score at the second stage)

  The Consistency model takes the feature vector from the first stage and converts it to a consistency score. This score measure the possibility that these two patches come from the same image.

#### Mask based approach:

Directly predict whether the image is sliced and where the sliced part is located.

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
#### EXIF based approach:

- ##### Feature Model

  Run the train_comparsion or train_prediction script to train the corresponding model.	

``` shell
python3 train_comparison.py
python3 train_prediction.py
```
- ##### Consistency Model
  Run the train_consistency script to train the Consistency Model.

``` shell
python3 train_consistency.py
```

#### Mask based approach:

```shell
python3 train_direct.py
```

## Evaluating

#### EXIF based approach:

Run the evaluation script to get the heatmap about the overall consistency scores of different areas in the given image.

``` shell
python3 test_consistency.py
```

#### Mask based approach:

```shell
python3 test_direct.py
```

