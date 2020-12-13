import requests
from pprint import pprint
import json
import os
import urllib
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import PIL.ExifTags 
# import exifread
# from google.colab import drive

'https://pixabay.com/api/?key=19093610-65af43f4077ea17750522df84&image_type=photo&pretty=true&per_page=200&page=21&category=backgrounds'


# download from each category until limit
# the visited_url_set is used to prevent duplicate pictures which appear in different category. 
def download(visited_url_set,category, max):
    counter = 1
    need_download_url = set()
    for cate in category:
        req_head = 'https://pixabay.com/api/?key=19093610-65af43f4077ea17750522df84&image_type=photo&pretty=true&per_page=200&page='
        req_end = '&category=' + cate
        print("start category " + cate)
        
        for i in range(1,3):
            page_index = str(i)
            new_end = page_index + req_end
            req_whole = req_head + new_end
            r = requests.get(req_whole)
            res = r.json()
            lists = res['hits']
            # if cate == 'backgrounds':
            # print(lists[0]['largeImageURL'])
            # print(lists[0]['id'])
            # print(lists[0]['pageURL'])
            
            print("set length", len(visited_url_set))
            for item in lists:
                originUrl = item['largeImageURL']
                if originUrl not in visited_url_set:
                    need_download_url.add(originUrl)
                    visited_url_set[originUrl] = 1
                # print(item['id'])
                # print(item['pageURL'])
                counter+=1
                if (counter >= max):
                    if os.path.exists("visited_url.txt"):
                        os.remove("visited_url.txt")
                    json.dump(visited_url_set, open("visited_url.txt",'w'))
                    return need_download_url
            print(counter)
    if os.path.exists("visited_url.txt"):
        os.remove("visited_url.txt")
    json.dump(visited_url_set, open("visited_url.txt",'w'))
    return need_download_url

# download a single url give a url, a destination folder_name, a boolean indicates if only download pictures contain exif
def download_single_url(url,folder_name,exifOnly):       
    target_folder = './cvFinalData/' + folder_name
    CHECK_FOLDER = os.path.isdir(target_folder)
    if not CHECK_FOLDER:
        os.makedirs(target_folder)
    current_total = len(os.listdir(target_folder))
    
    #name of the picture downloaded will be the current number of file
    final_path = target_folder + '/' + str(current_total) + '.jpg'
    # data = urllib.request.urlretrieve(url)
    print(url)
    try:
        r = requests.get(url,timeout=(7, 35))   # in case url takes too long to download
    except requests.exceptions.RequestException as e:
        print("download failed")
        with open('./cvFinalData/failToDownload.txt','a') as out:
            out.write(url+'\n')
        return
    print("url get succ")
    with open(final_path, 'wb') as outfile:
        outfile.write(r.content)

    # check if picture contains exif, if does, get exif and write to a txt file
    # if no exif found, delete the picture
    if exifOnly:
        image = Image.open(final_path)
        exifdata = image.getexif()
        if (exifdata != {}):
            print("#{} has exif data as #{}", current_total,exifdata)
            with open(target_folder+'.txt', 'a') as out:
                out.write(str(current_total)+' '+url + '\n')
            with open('exif_info'+'.txt', 'a') as out:
                out.write(str(current_total)+' '+str(exifdata) + '\n')
        else:
            print("#{} no exif found, removing #{}", final_path)
            os.remove(final_path)
    else:
        with open(target_folder+'.txt', 'a') as out:
            out.write(str(current_total)+' '+url + '\n')


def download_list_urls(URLs, folder_name, exifOnly):
    for i in URLs:
        download_single_url(i,folder_name,exifOnly)

def download_urls(max):
    # camera brand as tag, number as # of pictures want from tag
    category = ['backgrounds', 'fashion', 'nature', 'science', 'education', 'feelings', 'health', 'people', 'religion', 'places', 'animals', 'industry', 'computer', 'food', 'sports', 'transportation', 'travel', 'buildings', 'business', 'music']
    visitedUrl = {}
    if os.path.exists("visited_url.txt"):
        visitedUrl = json.load(open("visited_url.txt"))
    all_urls = download(visitedUrl, category, max)
    download_list_urls(all_urls, 'train', True)
        
download_urls(20000)  

