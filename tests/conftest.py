import sqlite3

import pytest
from werkzeug.security import generate_password_hash

from app import connect_db, create_app


@pytest.fixture()
def app():
    """Build an isolated Flask app instance with a dedicated temporary SQLite database."""
    test_app = create_app("testing")
    test_app.config["TESTING"] = True
    test_app.config["WTF_CSRF_ENABLED"] = False

    with test_app.app_context():
        conn = sqlite3.connect(test_app.config["DATABASE"], uri=False)
        conn.execute("CREATE TABLE IF NOT EXISTS analyzed_reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, review_text TEXT NOT NULL, sentiment TEXT NOT NULL, topic TEXT NOT NULL, aspects TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("CREATE TABLE IF NOT EXISTS info (id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT NOT NULL UNIQUE, name TEXT, email TEXT, mobile TEXT, password TEXT, role TEXT DEFAULT 'user', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("CREATE TABLE IF NOT EXISTS admin_users (id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT NOT NULL UNIQUE, name TEXT, email TEXT, mobile TEXT, password TEXT, role TEXT DEFAULT 'admin', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("DELETE FROM analyzed_reviews")
        conn.execute("DELETE FROM info")
        conn.execute("DELETE FROM admin_users")
        conn.execute(
            "INSERT INTO info (user, name, email, mobile, password, role) VALUES (?, ?, ?, ?, ?, ?)",
            ("admin", "Administrator", "admin@service.com", "0000000000", generate_password_hash("admin123"), "admin"),
        )
        conn.execute(
            "INSERT INTO admin_users (user, name, email, mobile, password, role) VALUES (?, ?, ?, ?, ?, ?)",
            ("admin", "Administrator", "admin@service.com", "0000000000", generate_password_hash("admin123"), "admin"),
        )
        conn.commit()
        conn.close()

    yield test_app

    with test_app.app_context():
        conn = sqlite3.connect(test_app.config["DATABASE"], uri=False)
        conn.execute("DELETE FROM analyzed_reviews")
        conn.execute("DELETE FROM info")
        conn.execute("DELETE FROM admin_users")
        conn.commit()
        conn.close()


@pytest.fixture()
def client(app):
    """Provide a Flask test client for request-based tests."""
    with app.test_client() as test_client:
        yield test_client


def seed_user(username="alice", password="StrongPass123", role="user", email="alice@example.com"):
    """Helper used by tests to insert a user into the app's info table."""
    conn = sqlite3.connect("file:test_app_db?mode=memory&cache=shared", uri=True)
    conn.execute(
        "INSERT INTO info (user, name, email, mobile, password, role) VALUES (?, ?, ?, ?, ?, ?)",
        (username, username.title(), email, "1234567890", generate_password_hash(password), role),
    )
    conn.commit()
    conn.close()
