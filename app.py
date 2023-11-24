from flask import Flask, render_template, request
import  webScraping as ws
# from mlmodel import Ratings
import pickle
import random

app = Flask(__name__)

# CourseInfo = {
#     'totalCourses': 0,
#     'courses': []
# }

@app.route('/')
def index():
    return render_template('index1.html')
@app.route('/index')
def index1():
    return render_template('index.html')

@app.route('/analysis',methods=['GET','POST'])
def analysis():
    if request.method == 'GET':
        return render_template('analysisForm.html')
    
    if request.method == 'POST':
        # call the SVM method of webScraping.py
        link = request.form.get('link')
        choose_model=request.form.get('platform')
        # site = request.form.get('platform')
        # val=ws.SVM(link)
        val=ws.Vishal(link,choose_model)
        # generate a random value for the rating between 0 to 1 
        # val=random.randint(0,1)
        print("val=",val)
        return render_template('analysisReport.html',val=val)

         
        

@app.route('/multipleAnalysis',methods=['GET','POST'])
# def multipleAnalysis():
#     if request.method == 'POST':
#         if request.form.get('platform'):
#             link = request.form.get('link')
#             site = request.form.get('platform')
#             CourseInfo['totalCourses'] += 1
#             v=CourseInfo['courses'].append(WebScraping(link,site))
#             print("chal Hat= ",v)
          
#             CourseInfo['courses'][-1]['rating'] = Ratings(CourseInfo['courses'][-1]['comments'])
#             CourseInfo['courses'][-1]['platform'] = site
#             CourseInfo['courses'][-1]['link'] = link
#         return render_template('multipleAnalysisReport.html',CourseInfo=CourseInfo)
    
#TODO
@app.route('/aboutus')
def aboutus():
    return render_template('aboutUs.html')

@app.route('/anim')
def aboutus1():
    return render_template('animation.html')

# if __name__ == "__main__":
app.run(debug=True)