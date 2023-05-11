from datetime import datetime
class PatientForm:
    def __init__(self,request) -> None:
        self.patient_Name = request.form['patient_name']
        self.patient_Id = request.form['patient_id']
        self.patient_age = request.form['patient_age']
        self.patient_gender = str(request.form['gender'])
        self.patient_type = request.form['patient_type']
        self.patient_loc = request.form['localization']


class PredictionHistory:
    def __init__(self,p_id,p_label,p_conf,p_all_conf,img) -> None:

        self.class_id = p_id
        self.class_label = p_label
        self.class_conf = p_conf
        self.all_class_conf = p_all_conf
        self.timestamp= datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.image_name = img
        
