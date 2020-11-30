import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from PIL.ExifTags import TAGS
import json
import os
import csv
def get_field (final_path,exif_data) :
    # print(final_path)
    image = Image.open(final_path)
    exif = image.getexif()
    for (k,v) in exif.items():
        fieldName = TAGS.get(k)
        exif_data[fieldName] = v

def create_whole_dict(fieldList, post_exif_data, fileName):
    res = {}
    for i in fieldList:
        res[i] = ""
    res["PictFilePath"] = fileName
    for key in post_exif_data:
        res[key] = post_exif_data[key]
    return res
# final_path = "./cvFinalData/test/0.jpg"

# get_field(exifdata, exif_data)
# print(exif_data) 

# file1 = open('./cvFinalData/test/exif_info.txt', 'r') 
# Lines = file1.readlines() 

def initial_csv(fieldList):
    with open('exif_table.csv', 'w', newline='') as csvfile:
        fieldnames = fieldList
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

fieldList = ["PictFilePath"]
for i in TAGS:
    fieldList.append(TAGS.get(i))  
initial_csv(fieldList)

for filename in os.listdir('./cvFinalData/test'):
    if not filename.startswith('.'):
        exif_data ={}
        get_field(os.path.join('./cvFinalData/test', filename), exif_data)
        exif_dict = create_whole_dict(fieldList,exif_data,filename)
        with open('exif_table.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldList)
            writer.writerow(exif_dict)
        print(exif_dict)



    
    


# exif = {
#     PIL.ExifTags.TAGS[k]: v
#     for k, v in exifdata.items():
#         if k in PIL.ExifTags.TAGS
# }


