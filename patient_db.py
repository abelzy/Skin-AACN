import pymysql
from datetime import datetime
from patient_info import PatientForm

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
        self.cursor.execute("CREATE TABLE IF NOT EXISTS patient_info (id INT AUTO_INCREMENT PRIMARY KEY, patient_id VARCHAR(255), patient_name VARCHAR(255), patient_age INT, patient_gender VARCHAR(10), patient_type VARCHAR(255), patient_loc VARCHAR(255), patient_label VARCHAR(255), date DATE)")
        self.disconnect()

    def insert_patient(self, patient_info):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.connect()
        self.cursor.execute(f"USE {self.database}")
        print(patient_info)
        self.cursor.execute("INSERT INTO patient_info (patient_id, patient_name, patient_age, patient_gender, patient_type, patient_loc, patient_label, date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                            (patient_info.patient_Id, patient_info.patient_Name, patient_info.patient_age, patient_info.patient_gender, patient_info.patient_type, patient_info.patient_loc, patient_info.patient_label, now))
        self.connection.commit()
        self.disconnect()

    def get_recent_patient_info(self):
        self.connect()
        self.cursor.execute(f"USE {self.database}")
        query = "SELECT patient_id, patient_name, patient_age, patient_gender, patient_type, patient_loc, patient_label FROM patient_info ORDER BY date DESC LIMIT 1"
        self.cursor.execute(query)
        patient = self.cursor.fetchone()
        return patient
