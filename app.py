from flask import Flask, render_template, request, redirect, url_for, session, Response
import csv
import json
from io import BytesIO, StringIO

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
app.secret_key = os.environ.get('SECRET_KEY', 'service-industry-admin-dashboard-secret')

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

    con.execute('''
        CREATE TABLE IF NOT EXISTS info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL UNIQUE,
            name TEXT,
            email TEXT,
            mobile TEXT,
            password TEXT,
            role TEXT DEFAULT 'user',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    info_columns = [row[1] for row in con.execute('PRAGMA table_info(info)')]
    if 'role' not in info_columns or 'created_at' not in info_columns:
        con.execute('ALTER TABLE info RENAME TO info_old')
        con.execute('''
            CREATE TABLE info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL UNIQUE,
                name TEXT,
                email TEXT,
                mobile TEXT,
                password TEXT,
                role TEXT DEFAULT 'user',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        selected_columns = ['id', 'user', 'name', 'email', 'mobile', 'password']
        if 'role' in info_columns:
            selected_columns.append('role')
        if 'created_at' in info_columns:
            selected_columns.append('created_at')
        column_sql = ', '.join(selected_columns)
        con.execute(f'''INSERT OR IGNORE INTO info ({column_sql})
            SELECT {column_sql} FROM info_old''')
        con.execute('DROP TABLE info_old')

    con.commit()
    con.close()


def ensure_default_admin():
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO info (user, name, email, mobile, password, role) VALUES (?, ?, ?, ?, ?, ?)",
        ('admin', 'Administrator', 'admin@service.com', '0000000000', 'admin123', 'admin')
    )
    con.commit()
    con.close()


init_analytics_db()
ensure_default_admin()


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
    global otp, username, email, number, password
    username = request.args.get('user','')
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
    global otp, username, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username, username, email, number, password))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signin.html")


@app.route("/signin", methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        mail1 = request.form.get('user', '').strip()
        password1 = request.form.get('password', '').strip()
    else:
        mail1 = request.args.get('user', '').strip()
        password1 = request.args.get('password', '').strip()

    if mail1 == 'admin' and password1 == 'admin123':
        session['user'] = mail1
        session['role'] = 'admin'
        return redirect(url_for('admin_dashboard'))

    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password`, `role` from info where `user` = ? AND `password` = ?", (mail1, password1,))
    data = cur.fetchone()
    con.close()

    if data is None:
        return render_template("signin.html", error='Invalid username or password.')

    if mail1 == str(data[0]) and password1 == str(data[1]):
        session['user'] = str(data[0])
        session['role'] = str(data[2] or 'user')
        return redirect(url_for('home'))

    return render_template("signin.html", error='Invalid username or password.')


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT user, password, role FROM info WHERE user = ? AND password = ?", (username, password))
        data = cur.fetchone()
        con.close()

        if data and username == str(data[0]) and password == str(data[1]) and str(data[2] or 'user') == 'admin':
            session['user'] = username
            session['role'] = 'admin'
            return redirect(url_for('admin_dashboard'))

        return render_template('admin_login.html', error='Invalid admin username or password.')

    return render_template('admin_login.html')


@app.route('/admin')
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))

    search = request.args.get('search', '').strip()
    role_filter = request.args.get('role', 'all').strip()

    con = sqlite3.connect('signup.db')
    con.row_factory = sqlite3.Row

    query = "SELECT id, user, name, email, mobile, role, created_at FROM info"
    params = []
    conditions = []

    if search:
        conditions.append("(user LIKE ? OR name LIKE ? OR email LIKE ? OR mobile LIKE ?)")
        search_term = f"%{search}%"
        params.extend([search_term, search_term, search_term, search_term])

    if role_filter != 'all':
        conditions.append("role = ?")
        params.append(role_filter)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY id DESC"
    users = con.execute(query, params).fetchall()

    registration_rows = con.execute(
        "SELECT substr(created_at, 1, 10) AS date, COUNT(*) AS count FROM info GROUP BY date ORDER BY date DESC LIMIT 14"
    ).fetchall()
    review_rows = con.execute(
        "SELECT substr(created_at, 1, 10) AS date, COUNT(*) AS count FROM analyzed_reviews GROUP BY date ORDER BY date DESC LIMIT 14"
    ).fetchall()
    sentiment_rows = con.execute(
        "SELECT sentiment, COUNT(*) AS count FROM analyzed_reviews GROUP BY sentiment ORDER BY count DESC"
    ).fetchall()
    con.close()

    registration_data = [
        {'date': row['date'], 'count': row['count']} for row in reversed(registration_rows)
    ]
    review_data = [
        {'date': row['date'], 'count': row['count']} for row in reversed(review_rows)
    ]
    sentiment_data = [
        {'label': row['sentiment'], 'count': row['count']} for row in sentiment_rows
    ]

    return render_template(
        'admin_dashboard.html',
        users=users,
        search=search,
        role_filter=role_filter,
        registration_rows=registration_data,
        review_rows=review_data,
        sentiment_rows=sentiment_data,
    )


@app.route('/admin/analytics-export')
def admin_analytics_export():
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))

    con = sqlite3.connect('signup.db')
    con.row_factory = sqlite3.Row

    total_users = con.execute('SELECT COUNT(*) AS count FROM info').fetchone()['count']
    total_reviews = con.execute('SELECT COUNT(*) AS count FROM analyzed_reviews').fetchone()['count']
    registration_rows = con.execute(
        "SELECT substr(created_at, 1, 10) AS date, COUNT(*) AS count FROM info GROUP BY date ORDER BY date DESC LIMIT 14"
    ).fetchall()
    review_rows = con.execute(
        "SELECT substr(created_at, 1, 10) AS date, COUNT(*) AS count FROM analyzed_reviews GROUP BY date ORDER BY date DESC LIMIT 14"
    ).fetchall()
    sentiment_rows = con.execute(
        "SELECT sentiment, COUNT(*) AS count FROM analyzed_reviews GROUP BY sentiment ORDER BY count DESC"
    ).fetchall()
    latest_reviews = con.execute(
        "SELECT review_text, sentiment, topic, created_at FROM analyzed_reviews ORDER BY id DESC LIMIT 8"
    ).fetchall()
    con.close()

    registration_data = [{'date': row['date'], 'count': row['count']} for row in reversed(registration_rows)]
    review_data = [{'date': row['date'], 'count': row['count']} for row in reversed(review_rows)]
    sentiment_data = [{'label': row['sentiment'], 'count': row['count']} for row in sentiment_rows]

    positive = sum(row['count'] for row in sentiment_rows if row['sentiment'] in ('Positive', 'Very Positive'))
    negative = sum(row['count'] for row in sentiment_rows if row['sentiment'] in ('Negative', 'Very Negative'))
    neutral = sum(row['count'] for row in sentiment_rows if row['sentiment'] in ('Neutral', 'Mixed'))

    return render_template(
        'admin_export.html',
        total_users=total_users,
        total_reviews=total_reviews,
        positive=positive,
        negative=negative,
        neutral=neutral,
        registration_rows=registration_data,
        review_rows=review_data,
        sentiment_rows=sentiment_data,
        latest_reviews=latest_reviews,
    )


@app.route('/admin/export-analytics-csv')
def export_analytics_csv():
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))

    con = sqlite3.connect('signup.db')
    con.row_factory = sqlite3.Row
    total_users = con.execute('SELECT COUNT(*) AS count FROM info').fetchone()['count']
    total_reviews = con.execute('SELECT COUNT(*) AS count FROM analyzed_reviews').fetchone()['count']
    registration_rows = con.execute(
        "SELECT substr(created_at, 1, 10) AS date, COUNT(*) AS count FROM info GROUP BY date ORDER BY date ASC"
    ).fetchall()
    sentiment_rows = con.execute(
        "SELECT sentiment, COUNT(*) AS count FROM analyzed_reviews GROUP BY sentiment ORDER BY sentiment ASC"
    ).fetchall()
    con.close()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['metric', 'value'])
    writer.writerow(['total_users', total_users])
    writer.writerow(['total_reviews', total_reviews])
    writer.writerow(['positive_reviews', sum(row['count'] for row in sentiment_rows if row['sentiment'] in ('Positive', 'Very Positive'))])
    writer.writerow(['negative_reviews', sum(row['count'] for row in sentiment_rows if row['sentiment'] in ('Negative', 'Very Negative'))])
    writer.writerow([])
    writer.writerow(['date', 'user_registrations'])
    for row in registration_rows:
        writer.writerow([row['date'], row['count']])
    writer.writerow([])
    writer.writerow(['sentiment', 'count'])
    for row in sentiment_rows:
        writer.writerow([row['sentiment'], row['count']])

    response = Response(output.getvalue(), mimetype='text/csv')
    response.headers['Content-Disposition'] = 'attachment; filename=admin_analytics_export.csv'
    return response


@app.route('/admin/edit/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))

    con = sqlite3.connect('signup.db')
    con.row_factory = sqlite3.Row
    user = con.execute("SELECT id, user, name, email, mobile, password, role FROM info WHERE id = ?", (user_id,)).fetchone()
    con.close()

    if not user:
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        username = request.form.get('user', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        mobile = request.form.get('mobile', '').strip()
        password = request.form.get('password', '').strip()
        role = request.form.get('role', 'user').strip()

        if not username:
            return render_template('admin_edit_user.html', user=dict(user), error='Username is required.')

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        if password:
            cur.execute(
                "UPDATE info SET user = ?, name = ?, email = ?, mobile = ?, password = ?, role = ? WHERE id = ?",
                (username, name, email, mobile, password, role, user_id)
            )
        else:
            cur.execute(
                "UPDATE info SET user = ?, name = ?, email = ?, mobile = ?, role = ? WHERE id = ?",
                (username, name, email, mobile, role, user_id)
            )
        con.commit()
        con.close()
        return redirect(url_for('admin_dashboard'))

    return render_template('admin_edit_user.html', user=dict(user))


@app.route('/admin/delete/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if session.get('role') != 'admin':
        return redirect(url_for('login'))

    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("SELECT user FROM info WHERE id = ?", (user_id,))
    user = cur.fetchone()

    if user and user[0] != 'admin':
        cur.execute("DELETE FROM info WHERE id = ?", (user_id,))
        con.commit()

    con.close()
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/export-excel')
def export_users_excel():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))

    from openpyxl import Workbook

    con = sqlite3.connect('signup.db')
    con.row_factory = sqlite3.Row
    users = con.execute(
        "SELECT id, user, name, email, mobile, role, created_at FROM info ORDER BY id DESC"
    ).fetchall()
    con.close()

    wb = Workbook()
    ws = wb.active
    ws.title = 'Users'
    ws.append(['id', 'user', 'name', 'email', 'mobile', 'role', 'created_at'])

    for row in users:
        ws.append([
            row['id'], row['user'], row['name'] or '', row['email'] or '',
            row['mobile'] or '', row['role'] or 'user', row['created_at'] or ''
        ])

    for col in ws.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except Exception:
                pass
        ws.column_dimensions[column].width = max_length + 2

    output = BytesIO()
    wb.save(output)
    output.seek(0)

    response = Response(output.getvalue(), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response.headers['Content-Disposition'] = 'attachment; filename=users.xlsx'
    return response


@app.route('/admin/raw-data')
def admin_raw_data():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))

    con = sqlite3.connect('signup.db')
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, user, name, email, mobile, password, role, created_at FROM info ORDER BY id DESC"
    ).fetchall()
    con.close()

    return render_template('admin_raw_data.html', rows=rows)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


BULK_BATCH_CACHE = {}

@app.route('/bulk_upload', methods=['GET', 'POST'])
def bulk_upload():
    if request.method == 'GET':
        return render_template('bulk_upload.html')
    
    file = request.files.get('file')
    if not file or not file.filename:
        return render_template('bulk_upload.html', error_msg='Please select a valid CSV or Excel file.')
    
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        if ext == '.csv':
            df = pd.read_csv(file)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file)
        else:
            return render_template('bulk_upload.html', error_msg='Unsupported file format. Please upload a .csv, .xlsx, or .xls file.')
    except Exception as e:
        return render_template('bulk_upload.html', error_msg=f'Error reading file: {str(e)}')
    
    if df.empty:
        return render_template('bulk_upload.html', error_msg='Uploaded file is empty.')
    
    candidate_cols = ['review', 'text', 'comment', 'feedback', 'review_text', 'comments', 'reviews', 'message']
    target_col = None
    for col in df.columns:
        if str(col).strip().lower() in candidate_cols:
            target_col = col
            break
    if not target_col:
        for col in df.columns:
            if df[col].dtype == object or isinstance(df[col].iloc[0], str):
                target_col = col
                break
        if not target_col:
            target_col = df.columns[0]
            
    processed_results = []
    db_records = []
    
    for _, row in df.iterrows():
        review_text = str(row[target_col]).strip() if pd.notna(row[target_col]) else ''
        if not review_text or len(review_text) < 2:
            continue
            
        vectorized = cv.transform([review_text])
        sentiment = classify_sentiment(review_text, vectorized)
        aspects = extract_aspects(review_text)
        
        try:
            df_single = pd.DataFrame({'sentence': [review_text]})
            _, word = Topic_modeling(df_single)
            topic = ', '.join(word) if (word and isinstance(word, list)) else 'General Services'
        except Exception:
            topic = 'General Services'
        
        processed_results.append({
            'review_text': review_text,
            'sentiment': sentiment,
            'aspects': aspects,
            'topic': topic,
        })
        
        db_records.append((review_text, sentiment, topic, json.dumps(aspects)))
        
    if db_records:
        con = sqlite3.connect('signup.db')
        con.executemany(
            'INSERT INTO analyzed_reviews (review_text, sentiment, topic, aspects) VALUES (?, ?, ?, ?)',
            db_records
        )
        con.commit()
        con.close()
        
    total_count = len(processed_results)
    if total_count == 0:
        return render_template('bulk_upload.html', error_msg='No valid text reviews found in file.')
        
    pos_count = sum(1 for r in processed_results if r['sentiment'] in ['Positive', 'Very Positive'])
    neg_count = sum(1 for r in processed_results if r['sentiment'] in ['Negative', 'Very Negative'])
    neu_count = total_count - pos_count - neg_count
    
    positive_pct = round((pos_count / total_count) * 100, 1)
    negative_pct = round((neg_count / total_count) * 100, 1)
    neutral_pct = round((neu_count / total_count) * 100, 1)
    
    BULK_BATCH_CACHE['last_batch'] = processed_results
    
    return render_template(
        'bulk_upload.html',
        results=processed_results[:100],
        total_count=total_count,
        positive_pct=positive_pct,
        negative_pct=negative_pct,
        neutral_pct=neutral_pct,
    )


@app.route('/download_sample_template')
def download_sample_template():
    sample_csv = "review_text\n\"The food was delicious and the server was amazingly quick!\"\n\"Room was dirty and wait time was unacceptable.\"\n\"Average product quality, nothing special.\"\n\"Fantastic customer support agent, solved my issue immediately.\"\n\"Packaging was damaged and delivery was delayed by 3 days.\"\n"
    response = Response(sample_csv, mimetype='text/csv')
    response.headers['Content-Disposition'] = 'attachment; filename=sample_reviews_template.csv'
    return response


@app.route('/download_bulk_export')
def download_bulk_export():
    batch = BULK_BATCH_CACHE.get('last_batch', [])
    fmt = request.args.get('format', 'csv').lower()
    
    if not batch:
        return redirect(url_for('bulk_upload'))
        
    export_data = []
    for item in batch:
        export_data.append({
            'Review Text': item['review_text'],
            'Sentiment': item['sentiment'],
            'Aspects': ', '.join(item['aspects']) if item['aspects'] else 'General',
            'Topic': item['topic']
        })
        
    df = pd.DataFrame(export_data)
    
    if fmt == 'excel':
        output = BytesIO()
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Bulk Analysis Results')
            output.seek(0)
            response = Response(output.getvalue(), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            response.headers['Content-Disposition'] = 'attachment; filename=bulk_sentiment_analysis_results.xlsx'
            return response
        except Exception:
            output = StringIO()
            df.to_csv(output, index=False)
            response = Response(output.getvalue(), mimetype='text/csv')
            response.headers['Content-Disposition'] = 'attachment; filename=bulk_sentiment_analysis_results.csv'
            return response
    else:
        output = StringIO()
        df.to_csv(output, index=False)
        response = Response(output.getvalue(), mimetype='text/csv')
        response.headers['Content-Disposition'] = 'attachment; filename=bulk_sentiment_analysis_results.csv'
        return response


@app.route('/notebook')
def notebook():
    return render_template('Notebook.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)