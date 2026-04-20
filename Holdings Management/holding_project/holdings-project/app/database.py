import sqlite3

def get_db():
    conn = sqlite3.connect("holdings.db", check_same_thread=False)
    return conn