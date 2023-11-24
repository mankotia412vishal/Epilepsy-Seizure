
import output_model as om

import numpy as np
def Vishal(link, platform):
    print("link=",link)
    print("platform=",platform)
    # call the Logistic function in output_model.py
    if platform=="Logistic" or platform=="Svm" or platform=="Knn" or platform=="Lstm" or platform=="Ann":

   
        return om.Logistic(link,platform)
    elif platform=="Svm":
        return om.Logistic(link,platform)
    elif platform=="Knn":
        return om.Logistic(link,platform)
    elif platform=="Lstm":
        return om.Logistic(link,platform)
    elif platform=="Ann":
        return om.Logistic(link,platform)
     

    



























"""from bs4 import BeautifulSoup
import urllib.request, json
import requests
import numpy as np
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def SVM(input_data):

    print(type(input_data))
    input_data=input_data.replace(","," ")
    input_data=[int(i) for i in input_data.split()]

    print(type(input_data))

    # a=input("Enter the link of the video: ")




    input_data = np.array([input_data])

# Reshape the input_data to be a 2D array with a single sample
    input_data = input_data.reshape(1, -1)

 

    # Load your SVM model from 'svm_model.pkl'
    model = joblib.load('svm_model.pkl')

    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(input_data)
    std_data = scaler.transform(input_data)

    # Make predictions
    prediction = model.predict(std_data)
    print(prediction)

    # if prediction[0] == 0:
         
    # #    return " The Person does not have Epileptic Disease "
    #     # print(" The Person does not have Epileptic Disease ")
    # else:
        # return  " The Person has Epileptic Disease "
        # print(" The Person has Epileptic Disease ")
    
    
    return prediction

    

def WebScraping(link, platform):

    link=link.replace(","," ")
    link=[int(i) for i in link.split()]

    print(type(link))

    print(link)
 
    SVM(link)

 
















































    site = platform
    c_url = link
    comments = []
    c = []
    result = {}
    if site == "Coursera":
        result['platform'] = 'Coursera'
        if '?' in c_url:
            c_url = c_url[:c_url.index("?")]
        url = c_url
        response = requests.get(url)
        htmlcontent = response.content
        soup = BeautifulSoup(htmlcontent, "html.parser")
        c = soup.findAll('h1', {'class': 'banner-title banner-title-without--subtitle m-b-0'})
        for i in c:
            result["title"] = i.text
        c = soup.findAll('div', {'class': 'content-inner'})
        for i in c:
            result["description"] = i.text
        c = soup.findAll('div', {'class': '_16ni8zai m-b-0 m-t-1s'})
        for i in c:
            result["duration"] = i.text
        c = soup.findAll('h3', {'class': 'instructor-name headline-3-text bold'})
        for i in c:
            result["instructor"] = i.text
        c = soup.findAll('div', {'class': '_1fpiay2'})
        for i in c:
            result["learner_count"] = i.text
        for k in range(1, 5):
            url = c_url + "/reviews?page=" + str(k)
            response = requests.get(url)
            htmlcontent = response.content
            soup = BeautifulSoup(htmlcontent, "html.parser")
            container = soup.findAll('div', {'class': 'rc-CML font-lg show-soft-breaks cml-cui'})
            for j in container:
                comments.append(j.text)
        result["comments"] = comments
        return result

    elif site =="Youtube":
        result['v'] = link[link.index('=')+1:]
        videoId = c_url[(c_url.index("=") + 1):]
        url = 'https://youtube.googleapis.com/youtube/v3/videos?part=contentDetails&id={}&key={}'.format(
            videoId, 'AIzaSyDEgmEzXQ7GCRXqAa8ctgc6jA50vZJLhR4')
        response = urllib.request.urlopen(url)
        data = response.read()
        data = json.loads(data)
        duration = data["items"][0]["contentDetails"]["duration"]
        result["duration"] = ''
        if duration.find('H')!=-1:
            result["duration"] = result["duration"] + (duration[duration.find('T') + 1:duration.find('H')] + ' Hr ')
        if duration.find('M')!=-1 and duration.find('H')!=-1:
            result["duration"] = result["duration"] + ' ' + (duration[duration.find('H') + 1:duration.find('M')] + ' Mins')
        elif duration.find('M')!=-1:
            result["duration"] = result["duration"] + ' ' +(duration[duration.find('T') + 1:duration.find('M')] + ' Mins')

        url = 'https://youtube.googleapis.com/youtube/v3/videos?part=statistics&id={}&key={}'.format(
            videoId, 'AIzaSyDEgmEzXQ7GCRXqAa8ctgc6jA50vZJLhR4')
        response = urllib.request.urlopen(url)
        data = response.read()
        data = json.loads(data)
        result["learner_count"] = data["items"][0]["statistics"]["viewCount"] + ' views'

        url = 'https://youtube.googleapis.com/youtube/v3/videos?part=snippet&id={}&key={}'.format(
            videoId, 'AIzaSyDEgmEzXQ7GCRXqAa8ctgc6jA50vZJLhR4')
        response = urllib.request.urlopen(url)
        data = response.read()
        data = json.loads(data)
        result["platform"] = "YouTube"
        result["title"] = data["items"][0]["snippet"]["title"]
        result["instructor"] = data["items"][0]["snippet"]["channelTitle"]
        result["description"] = data["items"][0]["snippet"]["description"]

        url = 'https://youtube.googleapis.com/youtube/v3/commentThreads?part=snippet&maxResults=100&videoId={}&key={}'.format(
            videoId, 'AIzaSyDEgmEzXQ7GCRXqAa8ctgc6jA50vZJLhR4')
        response = urllib.request.urlopen(url)
        data = response.read()
        data = json.loads(data)
        for i in range(len(data["items"])):
            if data['items'][i]['snippet']['topLevelComment']['snippet']['textDisplay'] is not None:
                comments.append(data['items'][i]['snippet']['topLevelComment']['snippet']['textDisplay'])
        result["comments"] = comments
        
        return result
    
    elif site == "Udemy":
        videoId = c_url[51:(c_url.index('?')) - 1]
        description_link="https://www.udemy.com/course/{}".format(videoId)
        url = 'https://www.udemy.com/api-2.0/courses/{}/'.format(videoId)
        response = urllib.request.urlopen(url)
        data = response.read()
        data = json.loads(data)
        result["platform"] = "Udemy"
        result["title"] = data["title"]
        result["instructor"] = data["visible_instructors"][0]["display_name"]
        response = requests.get(url)
        htmlcontent = response.content
        # soup = BeautifulSoup(htmlcontent, "html.parser")
        # c = soup.findAll('div', {'data-purpose': 'safely-set-inner-html:description:description'})
        result["description"] = 'N.A'
        result["duration"] = 'N.A'
        result["learner_count"] = 'N.A'


        for i in range(1, 10):
            url = 'https://www.udemy.com/api-2.0/courses/{}/reviews/?page={}'.format(videoId, i)
            response = urllib.request.urlopen(url)
            data = response.read()
            data = json.loads(data)
            for j in range(len(data["results"])):
                if data["results"][j]["content"]!='':
                    comments.append(data["results"][j]["content"])
        result["comments"] = comments
        return(result)
"""