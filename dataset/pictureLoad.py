import flickrapi
import urllib
import time
from PIL import Image
import PIL.ExifTags 
# import exifread
# from google.colab import drive
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import _thread
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
# drive.mount('/content/gdrive')

# Flickr api access key 
flickr=flickrapi.FlickrAPI('3b787960aa69f496ec4129e584bd0d68', '578ec97e39473798', cache=True)


# keyword = 'sky'
counter = 0 
print("key settled")

url_list = []
N_MAX = 60000
# KEYWORD = 'cat'


SIZES = ["url_o", "url_k", "url_h", "url_l"]
def get_flickr_generator(camera):
    size_extras = ','.join(SIZES)
    photos = flickr.walk(
    tags=camera,   # 搜索标签
    extras=size_extras, # 一张照片下载什么大小的
    media='photos',  # 保留
    # content_type=1, #for phtotos use 1
    # has_geo=1,   #是否有地理位置（也许能增加未被splice可能）
    per_page=5, # 
    sort='relevance')
    return photos

#size
# url_o: Original (4520 × 3229)
# url_k: Large 2048 (2048 × 1463)
# url_h: Large 1600 (1600 × 1143)
# url_l=: Large 1024 (1024 × 732)
# url_c: Medium 800 (800 × 572)
# url_z: Medium 640 (640 × 457)
# url_m: Medium 500 (500 × 357)
# url_n: Small 320 (320 × 229)
# url_s: Small 240 (240 × 171)
# url_t: Thumbnail (100 × 71)
# url_q: Square 150 (150 × 150)
# url_sq: Square 75 (75 × 75)

def get_single_url(photo):      #get a single url from generator    
    for i in range(len(SIZES)): #choose the biggest avaible size
        url = photo.get(SIZES[i])
        if url:
            return url
def get_url_list(max, camera):          #get a list(size as max) of urls from generator  
    photos = get_flickr_generator(camera)
    counter=0
    URLs = []
    for photo in photos:
        if counter < max:
            url = get_single_url(photo)  # get preffered size url
            if url:
                URLs.append(url)
                # print(photo)
                # exif = photo.getExif()
                # print(exif)
                counter += 1
            # if no url for the desired sizes then try with the next photo
        else:
            break
    return URLs, counter

def outputUrls(URLs, name):
    with open(name + '.txt', 'w') as f:
        for url in URLs:
            f.write("%s\n" % url)
def download_single_url(url,folder_name):       #name the file with the counts of files in dir
    target_folder = './cvFinalData/' + folder_name
    CHECK_FOLDER = os.path.isdir(target_folder)
    if not CHECK_FOLDER:
        os.makedirs(target_folder)
        # print("created folder : ", folder_name)
    current_total = len(os.listdir(target_folder))
    # current_total = 1+len([name for name in os.listdir('.') if os.path.isfile(name)])
    
    data = urllib.request.urlretrieve(url)
    image = Image.open(data[0]) 
    mgplot = plt.imshow(image)
    print(target_folder + '/' + str(current_total) + '.jpg')
    image.save(target_folder + '/' + str(current_total) + '.jpg', 'JPEG')
def download_list_urls(URLs, folder_name):
    for i in URLs:
        # print(folder_name)
        download_single_url(i,folder_name)
def download_urls():
    # camera brand as tag, number as # of pictures want from tag
    cameras = [['canon', 50], ['pentax',50], ['sony',50], ['nikon',50],['olympus',50]]
    all_urls = {} 
    
    time_counter = 0
    start = time.time()
    for i in range(len(cameras)):
        start = time.time()
        target_size = cameras[i][1]
        target_cam = cameras[i][0]
        URLs, tag_counter=get_url_list(target_size,target_cam)
        all_urls[target_cam] = URLs
        outputUrls(URLs, target_cam)    # print urls to a file
        print(target_cam, len(URLs))
        # try:    # start a new thread for download
        #     _thread.start_new_thread( download_list_urls,(URLs, target_cam))
        # except:
        #     print ("Error: unable to start thread")
        download_list_urls(URLs, target_cam)
        end = time.time()
        dif = end - start
        if time_counter > 3000:
            if dif < 3600:
                time.sleep(4000 - dif)
        start = time.time()
        time_counter = 0


if __name__=='__main__':
    # URLs, tag_counter=get_url_list(10)
    # print(len(URLs))

    # img = Image.open("./cvFinalData/1.JPG")
    # exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
    # print(exif)
    download_urls()

    # for i in URLs:
    #     print(i)
    #     download_single_url(i)
        # break

    # print("Start : %s" % time.ctime())
    # time.sleep( 10 )
    # print("End : %s" % time.ctime())

    # urllib.urlretrieve(urls[0], '00001.jpg')
    # data = urllib.request.urlretrieve(URLs[0])
    # print(data)
    # # Resize the image and overwrite it
    # image = Image.open(data[0]) 
    # image = image.resize((256, 256), Image.ANTIALIAS)
    # print(image)
    # imgplot = plt.imshow(image)
    # image.save('./cvFinalData/myphoto.jpg', 'JPEG')
