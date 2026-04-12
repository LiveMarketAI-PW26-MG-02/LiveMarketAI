from app.database import get_db

def create_table():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS holdings (
        instrument_id TEXT PRIMARY KEY,
        quantity INTEGER
    )
    """)

    conn.commit()