from __future__ import division, print_function
#Visualization and Image processing utils
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torch
#Pdf generator utils
from datetime import date
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
#database
from patient_db import PatientDatabase
# Flask utils
from flask import Flask, request, render_template,send_file
from werkzeug.utils import secure_filename
from model import convnextaa_base #AACN model
from patient_info import * # Pateint Form classes

# Defining the flask app
app_root = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder = os.path.abspath('static/'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "./static/model/conv_check.pt"
model = convnextaa_base(num_classes=7, pretrained=True, aa="ecanet",path=model_path)
model.to(device)
classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
        'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']
#initialize Database
db = PatientDatabase(database="skinaacn")
db.create_database()
db.create_tables()

def model_predict(input_img):
#Set the model for evaluation
    model.eval()
    output = model(input_img)
    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(output) 
    p= probabilities.cpu()
    num = p.detach().numpy()[0]
    per =[]
    for i in num:
        per.append(i*100)
    per =np.round_(per,2)
    return per


def loadImage(image_path):
    # Open the image using PIL
    image = Image.open(image_path).convert('RGB')
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image)
    # Add a batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = image_tensor.to(device)
    return image_tensor

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "GET":
        print("IN get")
        return render_template("index.html")

    if request.method == 'POST':
        image_file = request.files['myfile']
        global patient_info
        patient_info = PatientForm( request)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        img_data = "./static/lesions/" + secure_filename(patient_info.patient_Id) + ".jpg"
        image_file.save(img_data)
        input_img = loadImage(img_data)
        # Make prediction
        pred_probs = model_predict(input_img)
        # Map output to labels
        class_id = pred_probs.argmax()
        pred_class = classes[class_id]
        global prediction_info
        prediction_info = PredictionHistory(class_id,pred_class,pred_probs[class_id],pred_probs,img_data)
        save_Patient_info()
        lis = [classes,pred_probs]
        result = {"class":pred_class, "probs":lis, "image":img_data}
        return render_template('result2.html', result=result)

    return None

def save_Patient_info():
    # Create a bar chart of the confidence values
    x_labels = classes
    y_values = [float(c) for c in prediction_info.all_class_conf]
    fig, ax = plt.subplots()
    sns.barplot(x=x_labels, y=y_values, ax=ax)
    ax.set_ylim(0, 1)
    fig.subplots_adjust(bottom=0.45)
    ax.set_ylabel('Confidence')
    ax.set_title('Model Prediction')
    plt.xticks(rotation=45)
    # Save the bar chart to a file
    fig.savefig('chart.png')
    #save patient info into database
    db.insert_patient(patient_info)
    db.insert_classification(prediction_info)
    db.insert_prediction(patient_info,prediction_info)

    

@app.route('/report', methods=['POST'])
def generate_report():
    #  read data from database
    patinet_data_pdf =db.get_recent_db()
    str_patient_pdf = ["Patient ID: ","Patient Name: ","Patient Age: ",
                       "Patient Gender: ","Patient Type: ",
                       "Skin localization: ","Classification Result: "]
    # read chart image data from chart.png file
    with open("chart.png", "rb") as file:
        chart_data = file.read()
    # # Create a PDF buffer
    buffer = BytesIO()
    # # Create the PDF object, using the BytesIO object as its "file."
    pdf = canvas.Canvas(buffer, pagesize=letter)
    # # Insert the patient information into the PDF
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.black)
    y_pos = 10
    for i in range(len(patinet_data_pdf)):
        pdf.drawString(0.5*inch, y_pos*inch, str_patient_pdf[i] + str(patinet_data_pdf[i]))
        y_pos-=0.5
    # add the date to the report
    pdf.drawString(6.5 * inch, 10.5 * inch, date.today().strftime("%B %d, %Y"))
    # Insert the image into the PDF
    pdf.drawImage("input.jpg",5 * inch, 8 * inch, width=2.5 * inch, height=2 * inch)
    # add the chart image and result to the report
    chart_data = "chart.png"
    pdf.drawImage(chart_data, 1 * inch, 1 * inch, width=6.5 * inch, height=6 * inch)
    # Save the PDF to the buffer and close it
    pdf.save()
    buffer.seek(0)
    # Return the PDF file as a download
    return send_file(path_or_file=buffer, download_name=f'{patinet_data_pdf[0]}_report.pdf', as_attachment=True)

@app.route('/history', methods=['POST'])
def history():
    # get all prediction history and Info
    result= db.get_patient_history()
    return render_template('history4.html', result=result)

if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug = True, use_reloader=False)
