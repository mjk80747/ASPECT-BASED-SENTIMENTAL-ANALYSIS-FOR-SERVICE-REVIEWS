from flask import Flask, render_template, request, redirect, url_for, session, Response, current_app
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
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone, timedelta
from functools import wraps
import logging
import tempfile
import uuid
import nltk

from dotenv import load_dotenv
load_dotenv()

from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
import bleach

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

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", os.urandom(32).hex())
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = os.environ.get("FLASK_ENV") == "production"
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)
    DATABASE = os.environ.get("DATABASE_PATH", "signup.db")
    WTF_CSRF_ENABLED = True


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False
    DATABASE = os.environ.get("TEST_DATABASE_URI")


class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}


def create_app(config_name=None):
    """Create a Flask app configured for development, testing, or production."""
    app = Flask(__name__)
    chosen = config_by_name.get(config_name or os.environ.get("FLASK_ENV", "development"), DevelopmentConfig)
    app.config.from_object(chosen)

    if config_name == "testing" and not app.config.get("DATABASE"):
        fd, path = tempfile.mkstemp(prefix="sentiment_testing_", suffix=".db")
        os.close(fd)
        app.config["DATABASE"] = path
    else:
        app.config.setdefault("DATABASE", os.environ.get("DATABASE_PATH", "signup.db"))

    csrf.init_app(app)
    login_manager.init_app(app)
    limiter.init_app(app)

    @app.after_request
    def add_security_headers(response):
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
            "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
            "img-src 'self' data:;"
        )
        return response

    with app.app_context():
        init_analytics_db(app.config["DATABASE"])
        ensure_default_admin(app.config["DATABASE"])

    register_routes(app)
    globals()["app"] = app
    return app


def get_database_path(database_path=None):
    if database_path:
        return database_path
    try:
        return current_app.config.get("DATABASE", os.environ.get("DATABASE_PATH", "signup.db"))
    except RuntimeError:
        return os.environ.get("DATABASE_PATH", "signup.db")


def connect_db(database_path=None):
    target = get_database_path(database_path)
    conn = sqlite3.connect(target, uri=target.startswith("file:"))
    conn.row_factory = sqlite3.Row
    return conn


csrf = CSRFProtect()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("security")


class User(UserMixin):
    def __init__(self, id, user, name, email, mobile, role):
        self.id = str(id)
        self.user = user
        self.name = name
        self.email = email
        self.mobile = mobile
        self.role = role

login_manager = LoginManager()
login_manager.login_view = 'signin'

@login_manager.user_loader
def load_user(user_id):
    con = connect_db()
    row = con.execute("SELECT id, user, name, email, mobile, role FROM info WHERE id = ?", (user_id,)).fetchone()
    con.close()
    if row:
        return User(row['id'], row['user'], row['name'], row['email'], row['mobile'], row['role'] or 'user')
    return None

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or getattr(current_user, 'role', '') != 'admin':
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

limiter = Limiter(
    get_remote_address,
    default_limits=[],
    storage_uri="memory://"
)

BLOCKLIST_PASSWORDS = {'password', '12345678', '123456789', 'admin123', 'qwerty', 'password123', 'letmein'}

def validate_strong_password(password):
    if not password or len(password) < 10:
        return "Password must be at least 10 characters long."
    if password.lower() in BLOCKLIST_PASSWORDS:
        return "This password is too common and unsafe. Please choose a stronger password."
    if not re.search(r'[A-Z]', password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r'[a-z]', password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r'[0-9]', password):
        return "Password must contain at least one number."
    return None

def init_analytics_db(database_path=None):
    target_db = get_database_path(database_path)
    con = sqlite3.connect(target_db, uri=target_db.startswith("file:"))
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

    con.execute('''
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL UNIQUE,
            name TEXT,
            email TEXT,
            mobile TEXT,
            password TEXT,
            role TEXT DEFAULT 'admin',
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

    users_rows = con.execute("SELECT id, password FROM info").fetchall()
    for u_id, u_pass in users_rows:
        if u_pass and not (u_pass.startswith('scrypt:') or u_pass.startswith('pbkdf2:') or u_pass.startswith('argon2:')):
            hashed = generate_password_hash(u_pass)
            con.execute("UPDATE info SET password = ? WHERE id = ?", (hashed, u_id))

    con.commit()
    con.close()


def ensure_default_admin(database_path=None):
    target_db = get_database_path(database_path)
    con = sqlite3.connect(target_db, uri=target_db.startswith("file:"))
    cur = con.cursor()
    cur.execute("SELECT id, password FROM info WHERE user = ?", ('admin',))
    existing = cur.fetchone()
    if not existing:
        hashed_pass = generate_password_hash('admin123')
        cur.execute(
            "INSERT INTO info (user, name, email, mobile, password, role) VALUES (?, ?, ?, ?, ?, ?)",
            ('admin', 'Administrator', 'admin@service.com', '0000000000', hashed_pass, 'admin')
        )
    else:
        admin_pass = existing[1]
        if admin_pass and not (admin_pass.startswith('scrypt:') or admin_pass.startswith('pbkdf2:') or admin_pass.startswith('argon2:')):
            hashed_pass = generate_password_hash(admin_pass)
            cur.execute("UPDATE info SET password = ? WHERE id = ?", (hashed_pass, existing[0]))

    cur.execute("SELECT id FROM admin_users WHERE user = ?", ('admin',))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO admin_users (user, name, email, mobile, password, role) VALUES (?, ?, ?, ?, ?, ?)",
            ('admin', 'Administrator', 'admin@service.com', '0000000000', generate_password_hash('admin123'), 'admin')
        )

    con.commit()
    con.close()


# Keep the default runtime database consistent with the original single-file app.
init_analytics_db()
ensure_default_admin()


base_dir = os.path.dirname(os.path.abspath(__file__))
cv = pickle.load(open(os.path.join(base_dir, 'model.pickle'), 'rb'))
model = joblib.load(os.path.join(base_dir, 'model.sav'))

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

def index():
    return render_template("index.html")


def home():
    return render_template("home.html")


def about():
    return render_template('about.html')


def logon():
    return render_template('signin.html')


def login():
    return render_template('signin.html')


def upload():
    if request.method == 'GET':
        return render_template('home.html')

    raw_message = request.form.get('message', '').strip()
    if not raw_message:
        return render_template('home.html', prediction_error='Please enter a message to analyze.'), 400

    if len(raw_message) > 5000:
        return render_template('home.html', prediction_error='Message text is too long (maximum 5,000 characters).'), 400

    message = bleach.clean(raw_message, strip=True)

    data = [message]
   
    vect = cv.transform(data).toarray()
    sentiment = classify_sentiment(message, vect)
    aspects = extract_aspects(message)

    df = pd.DataFrame({'sentence':data})
    t,word = Topic_modeling(df)

    pred = f"{sentiment} Review, Based on the Input Message!"

    detected_topic = ', '.join(word) if word else 'Uncategorized'
    con = connect_db()
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


def analytics():
    con = connect_db()
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


@limiter.limit("3 per 5 minutes")
def signup():
    username = request.args.get('user', '').strip()
    email = request.args.get('email', '').strip()
    number = request.args.get('mobile', '').strip()
    password = request.args.get('password', '').strip()

    if not username or not email or not password:
        return render_template("signin.html", error="Username, Email, and Password are required.")

    pwd_err = validate_strong_password(password)
    if pwd_err:
        return render_template("signin.html", error=pwd_err)

    con = connect_db()
    cur = con.cursor()
    cur.execute("SELECT id FROM info WHERE LOWER(user) = LOWER(?) OR LOWER(email) = LOWER(?)", (username, email))
    if cur.fetchone():
        con.close()
        return render_template("signin.html", error="Username or email is already registered.")
    con.close()

    otp_code = random.randint(100000, 999999)

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
    msg.set_content("Your OTP is : " + str(otp_code))
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

    session['pending_signup'] = {
        'username': username,
        'email': email,
        'number': number,
        'password': generate_password_hash(password),
        'otp': otp_code,
        'expires_at': (datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()
    }

    return render_template("otp.html")


def otp():
    pending = session.get('pending_signup')
    if not pending:
        return render_template("signin.html", error="Session expired. Please try signing up again.")

    user_otp = request.form.get('message', '').strip()
    now_ts = datetime.now(timezone.utc).timestamp()

    if now_ts > pending.get('expires_at', 0):
        session.pop('pending_signup', None)
        return render_template("signin.html", error="OTP has expired. Please try signing up again.")

    if user_otp.isdigit() and int(user_otp) == pending.get('otp'):
        con = connect_db()
        cur = con.cursor()
        try:
            cur.execute(
                "INSERT INTO info (user, name, email, mobile, password) VALUES (?, ?, ?, ?, ?)",
                (pending['username'], pending['username'], pending['email'], pending['number'], pending['password'])
            )
            con.commit()
        except sqlite3.IntegrityError:
            con.close()
            session.pop('pending_signup', None)
            return render_template("signin.html", error="Username or email is already registered.")
        con.close()
        session.pop('pending_signup', None)
        return render_template("signin.html", success="Account created successfully! Please sign in.")

    return render_template("otp.html", error="Invalid OTP. Please try again.")


@limiter.limit("10 per minute")
def signin():
    if request.method == 'POST':
        login_input = request.form.get('user', '').strip()
        password_input = request.form.get('password', '').strip()
    else:
        login_input = request.args.get('user', '').strip()
        password_input = request.args.get('password', '').strip()

    if not login_input and not password_input:
        return render_template("signin.html")

    if not login_input or not password_input:
        return render_template("signin.html", error='Please provide both username/email and password.')

    con = connect_db()
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT id, user, name, email, mobile, password, role FROM info WHERE LOWER(user) = LOWER(?) OR LOWER(email) = LOWER(?)",
        (login_input, login_input)
    ).fetchone()
    con.close()

    if row and check_password_hash(row['password'], password_input):
        user_obj = User(row['id'], row['user'], row['name'], row['email'], row['mobile'], row['role'] or 'user')
        session.permanent = True
        login_user(user_obj, remember=True)
        session['user'] = user_obj.user
        session['role'] = user_obj.role
        logger.info(f"Successful user login for '{user_obj.user}' from IP {request.remote_addr}")
        if user_obj.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('home'))

    logger.warning(f"Failed user login attempt for '{login_input}' from IP {request.remote_addr}")
    return render_template("signin.html", error='Invalid username/email or password.')


@limiter.limit("5 per minute")
def admin_login():
    if request.method == 'POST':
        login_input = request.form.get('username', '').strip()
        password_input = request.form.get('password', '').strip()

        con = connect_db()
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT id, user, name, email, mobile, password, role FROM info WHERE (LOWER(user) = LOWER(?) OR LOWER(email) = LOWER(?)) AND role = 'admin'",
            (login_input, login_input)
        ).fetchone()
        con.close()

        if row and check_password_hash(row['password'], password_input):
            user_obj = User(row['id'], row['user'], row['name'], row['email'], row['mobile'], row['role'])
            session.permanent = True
            login_user(user_obj, remember=True)
            session['user'] = user_obj.user
            session['role'] = 'admin'
            logger.info(f"Successful admin login for '{user_obj.user}' from IP {request.remote_addr}")
            return redirect(url_for('admin_dashboard'))

        logger.warning(f"Failed admin login attempt for '{login_input}' from IP {request.remote_addr}")
        return render_template('admin_login.html', error='Invalid admin username/email or password.')

    return render_template('admin_login.html')


def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect('/admin/login')

    search = request.args.get('search', '').strip()
    role_filter = request.args.get('role', 'all').strip()

    con = connect_db()
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


def admin_analytics_export():
    if session.get('role') != 'admin':
        return redirect('/admin/login')

    con = connect_db()
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


def export_analytics_csv():
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))

    con = connect_db()
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


def edit_user(user_id):
    if session.get('role') != 'admin':
        return redirect('/admin/login')

    con = connect_db()
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

        con = connect_db()
        cur = con.cursor()
        if password:
            pwd_err = validate_strong_password(password)
            if pwd_err:
                con.close()
                return render_template('admin_edit_user.html', user=dict(user), error=pwd_err)
            hashed_p = generate_password_hash(password)
            cur.execute(
                "UPDATE info SET user = ?, name = ?, email = ?, mobile = ?, password = ?, role = ? WHERE id = ?",
                (username, name, email, mobile, hashed_p, role, user_id)
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


def delete_user(user_id):
    if session.get('role') != 'admin':
        return redirect('/admin/login')

    con = connect_db()
    cur = con.cursor()
    cur.execute("SELECT user FROM info WHERE id = ?", (user_id,))
    user = cur.fetchone()

    if user and user[0] != 'admin':
        cur.execute("DELETE FROM info WHERE id = ?", (user_id,))
        con.commit()

    con.close()
    return redirect(url_for('admin_dashboard'))


def export_users_excel():
    if session.get('role') != 'admin':
        return redirect('/admin/login')

    from openpyxl import Workbook

    con = connect_db()
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


def admin_raw_data():
    if session.get('role') != 'admin':
        return redirect('/admin/login')

    con = connect_db()
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, user, name, email, mobile, password, role, created_at FROM info ORDER BY id DESC"
    ).fetchall()
    con.close()

    return render_template('admin_raw_data.html', rows=rows)


def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login'))


BULK_BATCH_CACHE = {}

def bulk_upload():
    if request.method == 'GET':
        return render_template('bulk_upload.html')
    
    file = request.files.get('file')
    if not file or not file.filename:
        return render_template('bulk_upload.html', error_msg='Please select a valid CSV or Excel file.')
    
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in ['.csv', '.xlsx']:
        return render_template('bulk_upload.html', error_msg='Unsupported file format. Only .csv and .xlsx files are permitted.')

    try:
        if ext == '.csv':
            df = pd.read_csv(file)
        elif ext == '.xlsx':
            df = pd.read_excel(file)
    except Exception as e:
        logger.warning(f"Error parsing uploaded file '{filename}': {str(e)}")
        return render_template('bulk_upload.html', error_msg='Error reading file payload. Ensure the file is a valid CSV or Excel file.')
    
    if df.empty:
        return render_template('bulk_upload.html', error_msg='Uploaded file is empty.')
    
    if len(df) > 1000:
        return render_template('bulk_upload.html', error_msg='File contains too many rows (maximum 1,000 rows per batch upload allowed).')

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
        raw_review = str(row[target_col]).strip() if pd.notna(row[target_col]) else ''
        if not raw_review or len(raw_review) < 2:
            continue
            
        review_text = bleach.clean(raw_review[:5000], strip=True)
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
        con = connect_db()
        con.executemany(
            'INSERT INTO analyzed_reviews (review_text, sentiment, topic, aspects) VALUES (?, ?, ?, ?)',
            db_records
        )
        con.commit()
        con.close()
        logger.info(f"Processed bulk upload of {len(db_records)} records from file '{filename}' by user '{getattr(current_user, 'user', 'guest')}'")

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


def download_sample_template():
    sample_csv = "review_text\n\"The food was delicious and the server was amazingly quick!\"\n\"Room was dirty and wait time was unacceptable.\"\n\"Average product quality, nothing special.\"\n\"Fantastic customer support agent, solved my issue immediately.\"\n\"Packaging was damaged and delivery was delayed by 3 days.\"\n"
    response = Response(sample_csv, mimetype='text/csv')
    response.headers['Content-Disposition'] = 'attachment; filename=sample_reviews_template.csv'
    return response


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


def notebook():
    return render_template('Notebook.html')

def register_routes(app):
    app.add_url_rule('/', view_func=index, methods=['GET'])
    app.add_url_rule('/home', view_func=home, methods=['GET'])
    app.add_url_rule('/about', view_func=about, methods=['GET'])
    app.add_url_rule('/logon', view_func=logon, methods=['GET'])
    app.add_url_rule('/login', view_func=login, methods=['GET'])
    app.add_url_rule('/predict', view_func=upload, methods=['GET', 'POST'])
    app.add_url_rule('/analytics', view_func=analytics, methods=['GET'])
    app.add_url_rule('/signup', view_func=signup, methods=['GET', 'POST'])
    app.add_url_rule('/otp', view_func=otp, methods=['GET', 'POST'])
    app.add_url_rule('/signin', view_func=signin, methods=['GET', 'POST'])
    app.add_url_rule('/admin/login', view_func=admin_login, methods=['GET', 'POST'])
    app.add_url_rule('/admin', view_func=admin_dashboard, methods=['GET'])
    app.add_url_rule('/admin/analytics-export', view_func=admin_analytics_export, methods=['GET'])
    app.add_url_rule('/admin/export-analytics-csv', view_func=export_analytics_csv, methods=['GET'])
    app.add_url_rule('/admin/edit/<int:user_id>', view_func=edit_user, methods=['GET', 'POST'])
    app.add_url_rule('/admin/delete/<int:user_id>', view_func=delete_user, methods=['POST'])
    app.add_url_rule('/admin/raw-data', view_func=admin_raw_data, methods=['GET'])
    app.add_url_rule('/admin/export-users-excel', view_func=export_users_excel, methods=['GET'])
    app.add_url_rule('/logout', view_func=logout, methods=['GET'])
    app.add_url_rule('/bulk_upload', view_func=bulk_upload, methods=['GET', 'POST'])
    app.add_url_rule('/download_sample_template', view_func=download_sample_template, methods=['GET'])
    app.add_url_rule('/download_bulk_export', view_func=download_bulk_export, methods=['GET'])
    app.add_url_rule('/notebook', view_func=notebook, methods=['GET'])


app = create_app()


@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('home.html', prediction_error='File payload too large. Maximum allowed upload size is 5 MB.'), 413


@app.errorhandler(400)
def bad_request_error(error):
    return render_template('home.html', prediction_error='Bad request or invalid security CSRF token.'), 400


if __name__ == '__main__':
    is_debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    cert_path = os.environ.get('SSL_CERT_PATH', 'cert.pem')
    key_path = os.environ.get('SSL_KEY_PATH', 'key.pem')
    use_https = os.environ.get('USE_HTTPS', 'false').lower() == 'true'

    # Do not force self-signed certificates by default. Browsers reject them with
    # NET::ERR_CERT_AUTHORITY_INVALID, which is what caused the "Your connection is not private" warning.
    if use_https and os.path.exists(cert_path) and os.path.exists(key_path):
        app.run(host='0.0.0.0', port=5000, debug=is_debug, ssl_context=(cert_path, key_path))
    else:
        app.run(host='0.0.0.0', port=5000, debug=is_debug)