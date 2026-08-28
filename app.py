from flask import Flask, render_template, request
import json

import os
import numpy as np
import pandas as pd
import random
import sqlite3
import smtplib
from email.message import EmailMessage
from email.utils import parseaddr
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
            aspects TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    columns = [row[1] for row in con.execute('PRAGMA table_info(analyzed_reviews)')]
    if 'aspects' not in columns:
        con.execute("ALTER TABLE analyzed_reviews ADD COLUMN aspects TEXT NOT NULL DEFAULT '[]'")
    con.commit()
    con.close()


init_analytics_db()


cv = pickle.load(open('model.pickle','rb')) 
model = joblib.load('model.sav')

SENTIMENT_LABELS = [
    'Very Negative',
    'Negative',
    'Neutral',
    'Mixed',
    'Positive',
    'Very Positive',
]
POSITIVE_CUES = {
    'amazing', 'best', 'decent', 'excellent', 'fairly', 'fantastic',
    'fine', 'friendly', 'good', 'great', 'helpful', 'love', 'perfect',
    'quick', 'recommend', 'satisfied', 'smooth', 'useful', 'wonderful',
}
NEGATIVE_CUES = {
    'awful', 'bad', 'broken', 'complaint', 'delay', 'dirty',
    'late', 'poor', 'rude', 'slow', 'terrible', 'unhappy',
    'worst', 'disappointed',
}
VERY_POSITIVE_CUES = {'amazing', 'best', 'excellent', 'fantastic', 'perfect', 'wonderful'}
VERY_NEGATIVE_CUES = {'awful', 'broken', 'hate', 'terrible', 'worst', 'disappointed'}
NEUTRAL_CUES = {'average', 'okay', 'ordinary', 'standard', 'typical', 'usual'}
ASPECT_KEYWORDS = {
    'Product quality': {'quality', 'material', 'durable', 'defect', 'broken', 'design', 'feature'},
    'Price and value': {'price', 'cost', 'value', 'expensive', 'cheap', 'discount', 'offer'},
    'Delivery': {'delivery', 'deliver', 'shipping', 'arrived', 'arrival', 'late', 'package'},
    'Packaging': {'package', 'packaging', 'packed', 'box', '包装', 'damaged'},
    'Seller': {'seller', 'vendor', 'merchant', 'seller'},
    'Returns and refund': {'return', 'refund', 'replacement', 'exchange', 'cancel'},
    'Customer support': {'support', 'helpdesk', 'complaint', 'response', 'agent', 'staff', 'service'},
    'Food and taste': {'food', 'taste', 'flavor', 'meal', 'dish', 'menu', 'restaurant'},
    'Waiting time': {'wait', 'waiting', 'queue', 'speed', 'slow', 'quick', 'delay'},
    'Cleanliness': {'clean', 'cleanliness', 'dirty', 'hygiene', 'room'},
    'Ambience': {'ambience', 'atmosphere', 'environment', 'location', 'comfort'},
    'Booking and reliability': {'booking', 'reservation', 'reliable', 'appointment', 'schedule'},
}


def classify_sentiment(message, vectorized_message):
    probabilities = model.predict_proba(vectorized_message)[0]
    class_probabilities = dict(zip(model.classes_, probabilities))
    positive_probability = float(class_probabilities.get(1, 0))
    words = set(re.findall(r"[a-z']+", message.lower()))
    positive_cues = len(words & POSITIVE_CUES)
    negative_cues = len(words & NEGATIVE_CUES)
    neutral_cues = len(words & NEUTRAL_CUES)
    very_positive_cues = len(words & VERY_POSITIVE_CUES)
    very_negative_cues = len(words & VERY_NEGATIVE_CUES)

    if positive_cues and negative_cues:
        return 'Mixed'
    if neutral_cues and not positive_cues and not negative_cues:
        return 'Neutral'
    if very_positive_cues and not negative_cues:
        return 'Very Positive'
    if very_negative_cues and not positive_cues:
        return 'Very Negative'
    if positive_cues and not negative_cues:
        return 'Positive'
    if negative_cues and not positive_cues:
        return 'Negative'
    if positive_probability >= 0.85:
        return 'Very Positive'
    if positive_probability >= 0.60:
        return 'Positive'
    if positive_probability <= 0.15:
        return 'Very Negative'
    if positive_probability <= 0.40:
        return 'Negative'
    return 'Neutral'


def extract_aspects(message):
    words = set(re.findall(r"[a-z']+", message.lower()))
    detected = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if words & {keyword for keyword in keywords if keyword.isascii()}:
            detected.append(aspect)
    return detected

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
    sentiment = classify_sentiment(message, vect)
    aspects = extract_aspects(message)

    df = pd.DataFrame({'sentence':data})
    t,word = Topic_modeling(df)

    pred = f"{sentiment} Review, Based on the Input Message!"

    detected_topic = ', '.join(word) if word else 'Uncategorized'
    con = sqlite3.connect('signup.db')
    con.execute(
        'INSERT INTO analyzed_reviews (review_text, sentiment, topic, aspects) VALUES (?, ?, ?, ?)',
        (message, sentiment, detected_topic, json.dumps(aspects)),
    )
    con.commit()
    con.close()
    
    return render_template(
        'predict.html', pred_output=pred, message=message, to=t, wo=word,
        aspects=aspects,
    )


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
            SUM(CASE WHEN sentiment = 'Very Positive' THEN 1 ELSE 0 END) AS very_positive,
            SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) AS positive,
            SUM(CASE WHEN sentiment = 'Neutral' THEN 1 ELSE 0 END) AS neutral,
            SUM(CASE WHEN sentiment = 'Mixed' THEN 1 ELSE 0 END) AS mixed,
            SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) AS negative,
            SUM(CASE WHEN sentiment = 'Very Negative' THEN 1 ELSE 0 END) AS very_negative
        FROM analyzed_reviews
        GROUP BY topic
        ORDER BY COUNT(*) DESC
        LIMIT 8
    ''').fetchall()
    trend_rows = con.execute('''
        SELECT substr(created_at, 1, 10) AS day,
            SUM(CASE WHEN sentiment = 'Very Positive' THEN 1 ELSE 0 END) AS very_positive,
            SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) AS positive,
            SUM(CASE WHEN sentiment = 'Neutral' THEN 1 ELSE 0 END) AS neutral,
            SUM(CASE WHEN sentiment = 'Mixed' THEN 1 ELSE 0 END) AS mixed,
            SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) AS negative,
            SUM(CASE WHEN sentiment = 'Very Negative' THEN 1 ELSE 0 END) AS very_negative
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

    sentiment_counts = {label: 0 for label in SENTIMENT_LABELS}
    for row in sentiment_rows:
        if row['sentiment'] in sentiment_counts:
            sentiment_counts[row['sentiment']] = row['count']
    positive = sentiment_counts['Positive'] + sentiment_counts['Very Positive']
    negative = sentiment_counts['Negative'] + sentiment_counts['Very Negative']
    overall_sentiment = max(sentiment_counts, key=sentiment_counts.get) if total else 'No data'
    return render_template(
        'analytics.html',
        total=total,
        positive=positive,
        negative=negative,
        positive_percent=round(positive / total * 100) if total else 0,
        negative_percent=round(negative / total * 100) if total else 0,
        sentiment_labels=json.dumps(SENTIMENT_LABELS),
        sentiment_counts=json.dumps([sentiment_counts[label] for label in SENTIMENT_LABELS]),
        sentiment_labels_display=SENTIMENT_LABELS,
        sentiment_count_values=[sentiment_counts[label] for label in SENTIMENT_LABELS],
        neutral_mixed=sentiment_counts['Neutral'] + sentiment_counts['Mixed'],
        overall_sentiment=overall_sentiment,
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

    smtp_user = os.getenv('SMTP_USER', '').strip()
    smtp_password = os.getenv('SMTP_PASSWORD', '').strip()
    smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com').strip()
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    if not smtp_user or not smtp_password:
        app.logger.error('SMTP_USER and SMTP_PASSWORD must be configured')
        return render_template(
            "otp.html",
            error="Email delivery is not configured. Set SMTP_USER and SMTP_PASSWORD, then try again."
        ), 503
    if parseaddr(smtp_user)[1] != smtp_user or '@' not in smtp_user:
        app.logger.error('SMTP_USER must be a complete email address')
        return render_template(
            "otp.html",
            error="Email delivery is not configured correctly. SMTP_USER must be a complete email address."
        ), 503

    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = smtp_user
    msg['To'] = email
    
    
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(smtp_user, smtp_password)
            smtp.send_message(msg)
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