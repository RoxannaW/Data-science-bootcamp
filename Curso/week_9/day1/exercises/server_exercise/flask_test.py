import os, sys
from flask import Flask, render_template, redirect, request, jsonify, flash 
import time
import random
import json
from werkzeug.utils import secure_filename
import urllib.request


#root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#sys.path.append(root_path)
#print(sys.path)
#-> does not work anymore, says __file__ does not excist?

app = Flask(__name__, template_folder="C:\\Users\\Roxan\\OneDrive\\Documentos\\My_map_2\\Data-science-bootcamp\\Curso\\week_9\\day1\\exercises\\server_exercise\\templates"
)
UPLOAD_FOLDER = "C:\\Users\\Roxan\\OneDrive\\Documentos\\My_map_2\\Data-science-bootcamp\\Curso\\week_9\\day1\\exercises\\server_exercise\\uploads"

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'11111\n\]/'


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			flash('File successfully uploaded')
			return redirect('/')
			#return redirect(url_for('uploaded_file',filename=filename))
		else:
			flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
			return redirect(request.url)




def main():
    
	print("STARTING PROCESS")
    #print(os.path.dirname(__file__))
setting_file = "C:\\Users\\Roxan\\OneDrive\\Documentos\\My_map_2\\Data-science-bootcamp\\Curso\\week_9\\day1\\exercises\\server_exercise\\settings.json"
    

with open(setting_file, "r") as json_file_readed:
	json_readed = json.load(json_file_readed)

SERVER_RUNNING = json_readed["server_running"] 
    
if SERVER_RUNNING:
    DEBUG = json_readed["debug"]
    HOST = json_readed["host"]
    PORT_NUM = json_readed["port"]
    app.run(debug=DEBUG, host=HOST, port=PORT_NUM)
else:
    print("Server settings.json doesn't allow to start server. " + 
            "Please, allow it to run it.")
            
if __name__ == "__main__":
    main()