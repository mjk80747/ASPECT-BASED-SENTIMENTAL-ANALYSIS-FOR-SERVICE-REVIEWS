from flask import Flask, render_template, request
import json

import os
import numpy as np
import pandas as pd
import random
import sqlite3
import smtplib
from email.message import EmailMessage
from werkzeug.utils import secure_filename
import nltk

_required_nltk = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet',
    'omw-1.4': 'corpora/omw-1.4',
}

def init_nltk_data():
    for pkg, resource in _required_nltk.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(pkg, quiet=True)

init_nltk_data()

from topic_modelling import Topic_modeling

import pickle
import joblib
import re



app = Flask(__name__)

def init_analytics_db():
    con = sqlite3.connect('signup.db')
    con.execute('''
        CREATE TABLE IF NOT EXISTS analyzed_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            topic TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    con.commit()
    con.close()


init_analytics_db()


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
    if request.method == 'GET':
        return render_template('home.html')

    message = request.form.get('message', '').strip()
    if not message:
        return render_template('home.html', prediction_error='Please enter a message to analyze.'), 400

    data = [message]
   
    vect = cv.transform(data).toarray()
    result = int(model.predict(vect)[0])

    df = pd.DataFrame({'sentence':data})
    t,word = Topic_modeling(df)

    #result = model.predict(vectorized_text)[0]
    #         
    if result == 0:
        pred = "Negative Review, Based on the Input Message!"
        sentiment = "Negative"
    elif result == 1:
        pred = "Positive Review, Based on the Input Message!"    
        sentiment = "Positive"

    detected_topic = ', '.join(word) if word else 'Uncategorized'
    con = sqlite3.connect('signup.db')
    con.execute(
        'INSERT INTO analyzed_reviews (review_text, sentiment, topic) VALUES (?, ?, ?)',
        (message, sentiment, detected_topic),
    )
    con.commit()
    con.close()
    
    return render_template('predict.html', pred_output = pred, message=message, to = t, wo = word)


@app.route('/analytics')
def analytics():
    con = sqlite3.connect('signup.db')
    con.row_factory = sqlite3.Row
    total = con.execute('SELECT COUNT(*) FROM analyzed_reviews').fetchone()[0]
    sentiment_rows = con.execute(
        'SELECT sentiment, COUNT(*) AS count FROM analyzed_reviews GROUP BY sentiment'
    ).fetchall()
    topic_rows = con.execute(
        'SELECT topic, COUNT(*) AS count FROM analyzed_reviews GROUP BY topic ORDER BY count DESC LIMIT 8'
    ).fetchall()
    topic_sentiment_rows = con.execute('''
        SELECT topic,
            SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) AS positive,
            SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) AS negative
        FROM analyzed_reviews
        GROUP BY topic
        ORDER BY COUNT(*) DESC
        LIMIT 8
    ''').fetchall()
    trend_rows = con.execute('''
        SELECT substr(created_at, 1, 10) AS day,
            SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) AS positive,
            SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) AS negative
        FROM analyzed_reviews
        GROUP BY day
        ORDER BY day
    ''').fetchall()
    recent_rows = con.execute('''
        SELECT review_text, sentiment, topic, created_at
        FROM analyzed_reviews
        ORDER BY id DESC
        LIMIT 8
    ''').fetchall()
    con.close()

    sentiment_counts = {'Positive': 0, 'Negative': 0}
    for row in sentiment_rows:
        if row['sentiment'] in sentiment_counts:
            sentiment_counts[row['sentiment']] = row['count']

    positive = sentiment_counts['Positive']
    negative = sentiment_counts['Negative']
    return render_template(
        'analytics.html',
        total=total,
        positive=positive,
        negative=negative,
        positive_percent=round(positive / total * 100) if total else 0,
        negative_percent=round(negative / total * 100) if total else 0,
        topic_rows=json.dumps([dict(row) for row in topic_rows]),
        topic_sentiment_rows=json.dumps([dict(row) for row in topic_sentiment_rows]),
        trend_rows=json.dumps([dict(row) for row in trend_rows]),
        recent_rows=recent_rows,
    )


@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(100000, 999999)

    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    if not smtp_user or not smtp_password:
        app.logger.error('SMTP_USER and SMTP_PASSWORD must be configured')
        return render_template(
            "otp.html",
            error="Email delivery is not configured. Set SMTP_USER and SMTP_PASSWORD, then try again."
        ), 503

    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = smtp_user
    msg['To'] = email
    
    
    try:
        s = smtplib.SMTP('smtp.gmail.com', 587, timeout=20)
        s.starttls()
        s.login(smtp_user, smtp_password)
        s.send_message(msg)
        s.quit()
    except Exception as e:
        app.logger.exception("Could not send OTP email: %s", e)
        return render_template(
            "otp.html",
            error="We could not send the OTP. Check the SMTP settings and try again."
        ), 502
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