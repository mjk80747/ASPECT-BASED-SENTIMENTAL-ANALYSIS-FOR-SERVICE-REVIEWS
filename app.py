from flask import Flask, render_template, request

import os
import numpy as np
import pandas as pd
import random
import sqlite3
import smtplib 
from email.message import EmailMessage
from werkzeug.utils import secure_filename
from topic_modelling import Topic_modeling

import pickle
import joblib
import re



app = Flask(__name__)
 

cv = pickle.load(open('model.pickle','rb')) 
model = joblib.load('model.sav')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logon')
def logon():
	return render_template('signin.html')

@app.route('/login')
def login():
	return render_template('signin.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    message = request.form['message']
    data = [message]
   
    vect = cv.transform(data).toarray()
    result = model.predict(vect)

    df = pd.DataFrame({'sentence':data})
    t,word = Topic_modeling(df)

    #result = model.predict(vectorized_text)[0]
    #         
    if result == 0:
        pred = "Negative Review, Based on the Input Message!"
    elif result == 1:
        pred = "Positive Review, Based on the Input Message!"    
    
    return render_template('predict.html', pred_output = pred, message=message, to = t, wo = word)


@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(100000, 999999)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "myprojectstp@gmail.com"
    msg['To'] = email
    
    
    try:
        s =  smtplib.SMTP('smtp.gmail.com', 587)     
        s.starttls()
        s.login("myprojectstp@gmail.com", "paxgxdrhifmqcrzn")
        s.send_message(msg)
        s.quit()
    except Exception as e:
        print(f"Could not send email: {e}")
    return render_template("otp.html") 


@app.route('/otp', methods=['POST'])
def otp():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signin.html")


@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("signin.html")

@app.route('/notebook')
def notebook():
    return render_template('Notebook.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)