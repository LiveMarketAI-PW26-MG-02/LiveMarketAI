from app.database import get_db
from app.services import get_price

def add_holding(instrument_id, quantity):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT quantity FROM holdings WHERE instrument_id=?", (instrument_id,))
    row = cursor.fetchone()

    if row:
        new_qty = row[0] + quantity
        cursor.execute("UPDATE holdings SET quantity=? WHERE instrument_id=?", (new_qty, instrument_id))
    else:
        cursor.execute("INSERT INTO holdings VALUES (?, ?)", (instrument_id, quantity))

    conn.commit()


def get_holdings():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM holdings")
    rows = cursor.fetchall()

    result = []
    for instrument, qty in rows:
        price = get_price(instrument)
        value = price * qty

        result.append({
            "instrument": instrument,
            "quantity": qty,
            "price": price,
            "value": value
        })

    return result


def total_value():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM holdings")
    rows = cursor.fetchall()

    total = 0
    for instrument, qty in rows:
        total += get_price(instrument) * qty

    return total


def summary():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM holdings")
    rows = cursor.fetchall()

    total = 0
    distribution = {}

    for instrument, qty in rows:
        value = get_price(instrument) * qty
        total += value
        distribution[instrument] = value

    for k in distribution:
        distribution[k] = round((distribution[k] / total) * 100, 2)

    return {
        "total_value": total,
        "distribution": distribution
    }