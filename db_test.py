import pymysql

# Connect to the MySQL server
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='zedo2508',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

# Create a new database called 'mydatabase'
with conn.cursor() as cursor:
    cursor.execute('CREATE DATABASE IF NOT EXISTS skinaacn')

# Connect to the new database
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='zedo2508',
    db='skinaacn',
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

# Get a cursor object
cursor = conn.cursor()

# Define the patient information
patient_name = "John Doe"
patient_age = 45
patient_gender = "Male"
patient_diagnosis = "Skin Lesion"

# Create a SQL query to insert patient information into the database
sql = "INSERT INTO patients (name, age, gender, diagnosis) VALUES (%s, %s, %s, %s)"

# Execute the query with the patient information as parameters
cursor.execute(sql, (patient_name, patient_age, patient_gender, patient_diagnosis))

# Commit the changes to the database
conn.commit()

# Close the database connection
conn.close()
