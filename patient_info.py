class PatientForm:
    def __init__(self,p_name,p_id,p_age,p_gender,p_type,p_loc,p_label,p_conf,img) -> None:
        self.patient_Name = p_name
        self.patient_Id = p_id
        self.patient_age = p_age
        self.patient_gender = p_gender
        self.patient_type = p_type
        self.patient_loc = p_loc
        self.patient_label = p_label
        self.patient_conf = p_conf
        self.image_name = img
