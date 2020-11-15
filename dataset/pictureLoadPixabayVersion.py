import requests
from pprint import pprint
import json
import os
import urllib
import time
from PIL import Image
import PIL.ExifTags 
# import exifread
# from google.colab import drive
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
'https://pixabay.com/api/?key=19093610-65af43f4077ea17750522df84&image_type=photo&pretty=true&per_page=200&page=21&category=backgrounds'



def download(url_set,category, max):
    counter = 1
    for cate in category:
        req_head = 'https://pixabay.com/api/?key=19093610-65af43f4077ea17750522df84&image_type=photo&pretty=true&per_page=200&page='
        req_end = '&category=' + cate
        print("start category " + cate)
        
        for i in range(3):
            i += 1
            page_index = str(i)
            new_end = page_index + req_end
            req_whole = req_head + new_end
            r = requests.get(req_whole)
            res = r.json()
            lists = res['hits']
            if cate == 'backgrounds':
                print(lists[0]['largeImageURL'])
                print("set length", len(url_set))
            for item in lists:
                url_set.add(item['largeImageURL'])
                counter+=1
                if (counter >= max):
                    return url_set
            print(counter)
    url_set = set()
def download_single_url(url,folder_name):       #name the file with the counts of files in dir
    target_folder = './cvFinalData/' + folder_name
    CHECK_FOLDER = os.path.isdir(target_folder)
    if not CHECK_FOLDER:
        os.makedirs(target_folder)
        # print("created folder : ", folder_name)
    current_total = len(os.listdir(target_folder))
    # current_total = 1+len([name for name in os.listdir('.') if os.path.isfile(name)])
    
    final_path = target_folder + '/' + str(current_total) + '.jpg'
    # data = urllib.request.urlretrieve(url)

    r = requests.get(url)
    with open(final_path, 'wb') as outfile:
        outfile.write(r.content)

    # image = Image.open(data[0]) 
    # mgplot = plt.imshow(image)

    
    # image.save(target_folder + '/' + str(current_total) + '.jpg', 'JPEG')
def download_list_urls(URLs, folder_name):
    for i in URLs:
        # print(folder_name)
        download_single_url(i,folder_name)
def download_urls(max):
    # camera brand as tag, number as # of pictures want from tag
    category = ['backgrounds', 'people', 'animals', 'transportation', 'backgrbuildingsounds']
    max = 1500  # do not set max higher than 4000
    all_urls = download(set(), category, max)
    download_list_urls(all_urls, 'train')
        
download_urls(100)    