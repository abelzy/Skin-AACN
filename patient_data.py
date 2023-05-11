import pymysql

# Database connection
conn = pymysql.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    db="your_database"
)

class Patient:
    def __init__(self, patient_id, age, gender, patient_type, localization):
        self.patient_id = patient_id
        self.age = age
        self.gender = gender
        self.patient_type = patient_type
        self.localization = localization
        
    def insert(self):
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO Patient (patient_id, age, gender, type, localization) VALUES (%s, %s, %s, %s, %s)"
                cursor.execute(sql, (self.patient_id, self.age, self.gender, self.patient_type, self.localization))
                conn.commit()
        except:
            conn.rollback()

class Classification:
    def __init__(self, class_id, class_name):
        self.class_id = class_id
        self.class_name = class_name
        
    def insert(self):
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO Classification (class_id, class_name) VALUES (%s, %s)"
                cursor.execute(sql, (self.class_id, self.class_name))
                conn.commit()
        except:
            conn.rollback()

class Prediction:
    def __init__(self, prediction_id, patient_id, class_id, probability_score, timestamp):
        self.prediction_id = prediction_id
        self.patient_id = patient_id
        self.class_id = class_id
        self.probability_score = probability_score
        self.timestamp = timestamp
        
    def insert(self):
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO Prediction (prediction_id, patient_id, class_id, probability_score, timestamp) VALUES (%s, %s, %s, %s, %s)"
                cursor.execute(sql, (self.prediction_id, self.patient_id, self.class_id, self.probability_score, self.timestamp))
                conn.commit()
        except:
            conn.rollback()
