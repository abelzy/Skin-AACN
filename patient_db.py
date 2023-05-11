import pymysql
from datetime import datetime
from patient_info import *

class PatientDatabase:
    def __init__(self, host="localhost", user="root", password="zedo2508", database="skinaacn"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def connect(self,conn_type=None):
        if conn_type == "new":
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password
            )
        else:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database= self.database
            )
        self.cursor = self.connection.cursor()

    def disconnect(self):
        self.cursor.close()
        self.connection.close()

    def create_database(self):
        self.connect("new")
        self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        self.connection.commit()
        self.disconnect()
    def create_tables(self):
            self.connect("new")

            try:
                self.cursor.execute(f"USE {self.database}")
                self.cursor.execute('''CREATE TABLE IF NOT EXISTS Patient (
                            patient_id CHAR(10) PRIMARY KEY,
                            patient_name VARCHAR(50),
                            age INT,
                            gender VARCHAR(10),
                            type VARCHAR(20),
                            localization VARCHAR(50)
                        )''')
                self.cursor.execute("""CREATE TABLE Classification (
                                    class_id INT PRIMARY KEY,
                                    class_name VARCHAR(50)
                                    )""")
                self.cursor.execute("""CREATE TABLE Prediction (
                                    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
                                    patient_id CHAR(10),
                                    class_id INT,
                                    probability_score FLOAT,
                                    timestamp DATETIME,
                                    image_path TEXT,
                                    FOREIGN KEY (patient_id) REFERENCES Patient(patient_id),
                                    FOREIGN KEY (class_id) REFERENCES Classification(class_id)
                                    )""")
                print("Tables created successfully")
            except Exception as e:
                print(f"Error creating tables: {e}")
            
            self.connection.commit()
            self.disconnect()


    def insert_patient(self, patient_info):
        self.connect()
        self.cursor.execute(f"USE {self.database}")
        query = "INSERT INTO Patient (patient_id, patient_name, age, gender, type, localization) VALUES (%s,%s, %s, %s, %s, %s)"
        values = (patient_info.patient_Id,patient_info.patient_Name, 
                  patient_info.patient_age,patient_info.patient_gender, 
                  patient_info.patient_type, patient_info.patient_loc)
        self.cursor.execute(query, values)
        self.connection.commit()
        self.disconnect()
    
    def insert_classification(self, pred_info):
        self.connect()
        self.cursor.execute(f"USE {self.database}")
        query = "INSERT IGNORE INTO classification (class_id, class_name) VALUES (%s, %s)"
        values = (pred_info.class_id, pred_info.class_label)
        self.cursor.execute(query, values)
        self.connection.commit()
        self.disconnect()
        
    def insert_prediction(self, patient_info, pred_info):
        self.connect()
        self.cursor.execute(f"USE {self.database}")
        query = "INSERT INTO Prediction (patient_id, class_id, probability_score, timestamp, image_path) VALUES (%s, %s, %s, %s, %s)"
        values = (patient_info.patient_Id,pred_info.class_id, pred_info.class_conf, pred_info.timestamp, pred_info.image_name)
        self.cursor.execute(query, values)
        self.connection.commit()
        self.disconnect()

    def get_recent_db(self):
        self.connect()
        self.cursor.execute(f"USE {self.database}")
        query = """
            SELECT patient.patient_id, patient.patient_name, patient.age, patient.gender, patient.type, patient.localization, classification.class_name
            FROM Patient
            LEFT JOIN Prediction ON Patient.patient_id = Prediction.patient_id
            LEFT JOIN classification ON Prediction.class_id = classification.class_id
            ORDER BY Prediction.timestamp DESC
            """
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        self.disconnect()
        return result[0]
  
    def get_patient_history(self):

        self.connect()
        self.cursor.execute(f"USE {self.database}")
        query = """
            SELECT patient.patient_id, patient.patient_name, patient.age, patient.gender, patient.type, patient.localization, prediction.probability_score, classification.class_name, prediction.timestamp
            FROM Patient
            LEFT JOIN Prediction ON Patient.patient_id = Prediction.patient_id
            LEFT JOIN classification ON Prediction.class_id = classification.class_id
            ORDER BY Prediction.timestamp DESC
            """
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        self.disconnect()
        return results




