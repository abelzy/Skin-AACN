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
# Flask utils
from flask import Flask, request, render_template,send_file
from werkzeug.utils import secure_filename
from model import convnextaa_base #AACN model
from patient_info import PatientForm # Pateint Form class

# Defining the flask app
app = Flask(__name__, static_folder = os.path.abspath('static/'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "./static/model/conv_check.pt"
model = convnextaa_base(num_classes=7, pretrained=True, aa="ecanet",path=model_path)
model.to(device)
classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
        'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']

def model_predict(input_img):

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
    print(per)

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
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        # image_location = os.path.join(basepath, 'static', secure_filename(image_file.filename))
        image_location = "input.jpg"
        image_file.save(image_location)
        input_img = loadImage(image_location)
        # Make prediction
        pred_probs = model_predict(input_img)
        # Map output to labels
        pred_class = classes[pred_probs.argmax()]
        img_data = "../static/"+secure_filename(image_file.filename)
        # img_data = "../input.jpg"
        global patient_info
        patient_info = PatientForm( request.form['patient_name'],request.form['patient_id'],request.form['patient_age'], str(request.form['gender']), request.form['patient_type'],request.form['localization'],pred_class,pred_probs,image_location)
        save_Patient_info(classes)
        lis = [classes,pred_probs]
        result = {"class":pred_class, "probs":lis, "image":img_data}
        return render_template('result2.html', result=result)

    return None

def save_Patient_info(class_labels):

    with open('result.txt', 'w') as f:
        f.write(f'Patient ID: {patient_info.patient_Id}\n')
        f.write(f'Patient Name: {patient_info.patient_Name}\n')
        f.write(f'Patient Age: {patient_info.patient_age}\n')
        f.write(f'Patient Gender: {patient_info.patient_gender}\n')
        f.write(f'Patient Type: {patient_info.patient_type}\n')
        f.write(f'Patient Localization: {patient_info.patient_loc}\n')
        f.write(f'Predicted Label: {patient_info.patient_label}\n')
        f.write(f'Image Name: {patient_info.image_name}\n')

    # Create a bar chart of the confidence values
    x_labels = class_labels
    y_values = [float(c) for c in patient_info.patient_conf]
    fig, ax = plt.subplots()
    sns.barplot(x=x_labels, y=y_values, ax=ax)
    ax.set_ylim(0, 1)
    fig.subplots_adjust(bottom=0.45)
    ax.set_ylabel('Confidence')
    ax.set_title('Model Prediction')
    plt.xticks(rotation=45)
    # Save the bar chart to a file
    fig.savefig('chart.png')


@app.route('/report', methods=['POST'])
def generate_report():
     # read data from result.txt file
    with open("result.txt", "r") as file:
        result_data = file.read().split("\n")
    patinet_data_pdf =[]
    str_patient_pdf = ["Patient ID: ","Patient Name: ","Patient Age: ","Patient Type: ","Skin localization: ","Classification Result: "]
    # extract patient information
    for i in len(result_data):
        patinet_data_pdf[i] = result_data[i].split(":")[1].strip()
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
    pos_i = 0
    for i in len(patinet_data_pdf)-1:
        pdf.drawString(0.5*inch, (y_pos+pos_i)*inch, str_patient_pdf[i] + patinet_data_pdf[i])
        pos_i+=0.5
    # add the date to the report
    pdf.drawString(6.5 * inch, 10.5 * inch, date.today().strftime("%B %d, %Y"))
    # Insert the image into the PDF
    pdf.drawImage(patinet_data_pdf[7],5 * inch, 8 * inch, width=2.5 * inch, height=2 * inch)
    # add the chart image and result to the report
    chart_data = "chart.png"
    pdf.drawImage(chart_data, 1 * inch, 1 * inch, width=6.5 * inch, height=6 * inch)
    # Save the PDF to the buffer and close it
    pdf.save()
    buffer.seek(0)
    # Return the PDF file as a download
    return send_file(path_or_file=buffer, download_name='classification_report.pdf', as_attachment=True)



if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug = True, use_reloader=False)
