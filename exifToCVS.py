from PIL import Image
from PIL.ExifTags import TAGS
import os
import csv


def initial_csv(fieldList):
    with open('temp/exif_table.csv', 'w', newline='') as csvfile:
        fieldnames = fieldList
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def get_exif_for_one_image (final_path, exif_dicts, file_name) :
    # open image
    image = Image.open(final_path)
    exif = image.getexif()
    exif_fields_fetched = []
    exif_dict = {}

    # check whether contains all necessary fields
    for (k, v) in exif.items():
        exif_fields_fetched.append(TAGS.get(k))
    for must in ['Make', 'Model', 'ExposureTime', 'ExifOffset', 'ISOSpeedRatings', 'FocalLength']:
        if must not in exif_fields_fetched:
            print('Not found', must)
            return

    # add all fields to the dict
    for (k, v) in exif.items():
        key = TAGS.get(k)
        if key in ['Make', 'Model', 'ExposureTime', 'ExifOffset', 'ISOSpeedRatings', 'FocalLength']:
            if key == 'ISOSpeedRatings' and str(v)[0] == '(':
                exif_dict[key] = int(str(v).split(',')[0][1:])
            else:
                exif_dict[key] = v
            values[key].append(exif_dict[key])

    exif_dict["PictFilePath"] = file_name
    exif_dicts.append(exif_dict)


def getPartitionID(list, value, partition_num):
    for i in range (1, partition_num + 1):
        index = min(len(list)-1, i * len(list) // partition_num)
        if list[index] >= value:
            return i - 1


def classify(exif_dicts, partition):
    # sort all value lists
    for (key, list) in values.items():
        list.sort()
        print(values[key])

    for i in range (0, len(exif_dicts)):
        for key in ['Make', 'Model', 'ExposureTime', 'ExifOffset', 'ISOSpeedRatings', 'FocalLength']:
            exif_dicts[i][key] = getPartitionID(values[key], exif_dicts[i][key], partition)


fieldList = ['PictFilePath', 'Make', 'Model', 'ExposureTime', 'ExifOffset', 'ISOSpeedRatings', 'FocalLength']
initial_csv(fieldList)

exif_dicts =[]
values = {
    'Make': [],
    'Model': [],
    'ExposureTime': [],
    'ExifOffset': [],
    'ISOSpeedRatings': [],
    'FocalLength': []
}

# fill in exif dicts and values
for filename in os.listdir('temp/images'):
    if not filename.startswith('.'):
        get_exif_for_one_image(os.path.join('temp/images', filename), exif_dicts, filename)

# classify
classify(exif_dicts, 10)

# write to excel file
for exif_dict in exif_dicts:
    with open('temp/exif_table.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldList)
        writer.writerow(exif_dict)
    print(exif_dict)
