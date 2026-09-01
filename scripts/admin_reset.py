#!/usr/bin/env python3
"""
Offline Admin Management & Database Backup Utility Script
Aspect-Based Sentiment Analysis Application
"""

import sys
import os
import sqlite3
import shutil
from datetime import datetime
from werkzeug.security import generate_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'signup.db')
BACKUP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backups')

def backup_database():
    if not os.path.exists(DB_PATH):
        print(f"[Error] Database file '{DB_PATH}' not found!")
        return None
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(BACKUP_DIR, f"signup_backup_{timestamp}.db")
    shutil.copy2(DB_PATH, dest)
    print(f"[Success] Created database backup at: {dest}")
    return dest

def reset_admin_password(new_password):
    if len(new_password) < 10:
        print("[Error] New admin password must be at least 10 characters long!")
        return False
    hashed = generate_password_hash(new_password)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id FROM info WHERE user = ?", ('admin',))
    row = cur.fetchone()
    if row:
        cur.execute("UPDATE info SET password = ? WHERE id = ?", (hashed, row[0]))
        print(f"[Success] Password for admin user (ID: {row[0]}) updated successfully!")
    else:
        cur.execute(
            "INSERT INTO info (user, name, email, mobile, password, role) VALUES (?, ?, ?, ?, ?, ?)",
            ('admin', 'Administrator', 'admin@service.com', '0000000000', hashed, 'admin')
        )
        print("[Success] Created new admin user account with hashed password!")
    con.commit()
    con.close()
    return True

def list_users():
    if not os.path.exists(DB_PATH):
        print("[Error] Database not found!")
        return
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    users = con.execute("SELECT id, user, name, email, role, created_at FROM info").fetchall()
    con.close()
    print("\n--- Current Users in Database ---")
    for u in users:
        print(f"ID: {u['id']} | User: {u['user']} | Email: {u['email']} | Role: {u['role']} | Created: {u['created_at']}")
    print("---------------------------------\n")

def main():
    print("=== Admin Security & Backup Utility ===")
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/admin_reset.py backup")
        print("  python scripts/admin_reset.py list")
        print("  python scripts/admin_reset.py reset-admin <new_password>")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "backup":
        backup_database()
    elif cmd == "list":
        list_users()
    elif cmd == "reset-admin":
        if len(sys.argv) < 3:
            print("[Error] Please provide the new admin password.")
            sys.exit(1)
        backup_database()
        reset_admin_password(sys.argv[2])
    else:
        print(f"[Error] Unknown command: {cmd}")

if __name__ == '__main__':
    main()
