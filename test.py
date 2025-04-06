import mysql.connector

try:
    connection = mysql.connector.connect(
        host="localhost",
        user="Traffic1",
        password="pedestrian",
        database="traffic_db"
    )
    if connection.is_connected():
        print("Successfully connected to MySQL database!")
        cursor = connection.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        print(f"MySQL version: {version[0]}")
except mysql.connector.Error as e:
    print(f"Error connecting to MySQL: {e}")
finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection closed.")