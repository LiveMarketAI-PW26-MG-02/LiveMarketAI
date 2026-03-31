
import mysql.connector
def log():
    try:
        db=mysql.connector.connect(host="localhost",user="root",password="")
        cursor=db.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS stock_memory")
        cursor.execute("USE stock_memory")
        cursor.execute("CREATE TABLE IF NOT EXISTS logs(val FLOAT)")
        cursor.execute("INSERT INTO logs(val) VALUES(1.0)")
        db.commit()
    except:
        pass
