import pymysql

class PatientDatabase:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def connect(self):
        self.connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password
        )
        self.cursor = self.connection.cursor()

    def disconnect(self):
        self.cursor.close()
        self.connection.close()

    def create_database(self):
        self.connect()
        self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        self.disconnect()

    def create_table(self):
        self.connect()
        self.cursor.execute(f"USE {self.database}")
        self.cursor.execute("CREATE TABLE IF NOT EXISTS patients (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), age INT, lesion_image VARCHAR(255), prediction VARCHAR(255))")
        self.disconnect()

    def insert_patient(self, patient_name, patient_age, lesion_image, prediction):
        self.connect()
        self.cursor.execute(f"USE {self.database}")
        query = "INSERT INTO patients (name, age, lesion_image, prediction) VALUES (%s, %s, %s, %s)"
        values = (patient_name, patient_age, lesion_image, prediction)
        self.cursor.execute(query, values)
        self.connection.commit()
        self.disconnect()
