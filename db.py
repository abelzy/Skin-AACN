import pymysql

# Connect to the MySQL server
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='password',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

# Create a new database called 'mydatabase'
with conn.cursor() as cursor:
    cursor.execute('CREATE DATABASE IF NOT EXISTS mydatabase')

# Connect to the new database
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='password',
    db='mydatabase',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

# Create a new 'patients' table
with conn.cursor() as cursor:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            age INT NOT NULL,
            gender VARCHAR(10) NOT NULL,
            diagnosis VARCHAR(255) NOT NULL
        )
    ''')

# Commit the changes to the database
conn.commit()

# Close the database connection
conn.close()
