import mysql.connector
from mysql.connector import Error

class SkinLesionDatabase:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.connection.cursor()
            print("Connection established")
        except Error as e:
            print(f"The error '{e}' occurred")

    def create_patient_info_table(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS patient_info (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), age INT, sex VARCHAR(10), date DATE, lesion_image BLOB)")
        print("Table created")

    def insert_patient_info(self, name, age, sex, date, lesion_image):
        query = "INSERT INTO patient_info (name, age, sex, date, lesion_image) VALUES (%s, %s, %s, %s, %s)"
        values = (name, age, sex, date, lesion_image)
        self.cursor.execute(query, values)
        self.connection.commit()
        print(f"{self.cursor.rowcount} record inserted.")

    def get_recent_patient_info(self):
        query = "SELECT * FROM patient_info ORDER BY date DESC LIMIT 1"
        self.cursor.execute(query)
        result = self.cursor.fetchone()
        return result

    def close(self):
        self.connection.close()
        print("Connection closed")
