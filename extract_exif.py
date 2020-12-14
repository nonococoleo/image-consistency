import os
import csv
from PIL import Image
from PIL.ExifTags import TAGS


def get_exif_for_one_image(file_path):
    """
    extract exif from image
    :param file_path: path to image file
    :return: exif
    """

    # open image
    image = Image.open(file_path)
    exif = image.getexif()
    exif_fields_fetched = []
    exif_dict = {}

    # check whether contains all necessary fields
    for (k, v) in exif.items():
        exif_fields_fetched.append(TAGS.get(k))
    for must in ['Make', 'Model', 'ExposureTime', 'ExifOffset', 'ISOSpeedRatings', 'FocalLength']:
        if must not in exif_fields_fetched:
            print('Not found', must)
            return None

    # add all fields to the dict
    for (k, v) in exif.items():
        key = TAGS.get(k)
        if key in ['Make', 'Model', 'ExposureTime', 'ExifOffset', 'ISOSpeedRatings', 'FocalLength']:
            if key == 'ISOSpeedRatings' and str(v)[0] == '(':
                exif_dict[key] = int(str(v).split(',')[0][1:])
            else:
                exif_dict[key] = v
            values[key].append(exif_dict[key])

    return exif_dict


def get_partition_ID(li, value, num_partition):
    """
    get partition id for current value
    :param li: list of values
    :param value: current value
    :param num_partition: num of partition
    :return: partition id for current value
    """

    for i in range(1, num_partition + 1):
        index = min(len(li) - 1, i * len(li) // num_partition)
        if li[index] >= value:
            return i - 1


if __name__ == '__main__':
    rootDir = "datasets/exif/"
    fileName = rootDir + 'train.csv'
    fieldList = ['PictFilePath', 'Make', 'Model', 'ExposureTime', 'ExifOffset', 'ISOSpeedRatings', 'FocalLength']
    with open(fileName, 'w', newline='') as csvfile:
        fieldnames = fieldList
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    exif_dicts = []
    values = {
        'Make': [],
        'Model': [],
        'ExposureTime': [],
        'ExifOffset': [],
        'ISOSpeedRatings': [],
        'FocalLength': []
    }

    # fill in exif dicts and values
    for filename in os.listdir(rootDir + 'images'):
        if not filename.startswith('.'):
            exif = get_exif_for_one_image(os.path.join(rootDir + 'images', filename))
            if exif is not None:
                exif['PictFilePath'] = filename
                exif_dicts.append(exif)

    # classify
    # sort all value lists
    for (key, li) in values.items():
        li.sort()
        print(values[key])

    for i in range(0, len(exif_dicts)):
        for key in ['Make', 'Model', 'ExposureTime', 'ExifOffset', 'ISOSpeedRatings', 'FocalLength']:
            exif_dicts[i][key] = get_partition_ID(values[key], exif_dicts[i][key], num_partition=10)

    # write to excel file
    for exif_dict in exif_dicts:
        with open(fileName, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldList)
            writer.writerow(exif_dict)
        print(exif_dict)
