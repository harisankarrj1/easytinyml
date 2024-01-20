from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import subprocess
from flask import send_file
from tensorflow import keras
import edgeimpulse as ei
import tensorflow as tf

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


ei.API_KEY = "ei_de71dadcd64f64e06980b32bc1b7c4222f102dd6e3cf373f" 

app = Flask(__name__)

# List to store labels
labels_list = []

deploy_filename = ""
deploy_target = ""

# Variable to store the path of the uploaded model
uploaded_model_path = ""


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    label = request.form['label']
    upload_folder = os.path.join(os.getcwd(), 'data', label)
    os.makedirs(upload_folder, exist_ok=True)

    for file in request.files.getlist('images'):
        filename = secure_filename(file.filename)
        file.save(os.path.join(upload_folder, filename))

    return 'Upload successful!'

@app.route('/train_model', methods=['GET'])
def train_model():
    try:
        # Run the train.py script
        result = subprocess.run(['python', 'train.py'], capture_output=True, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"

    except Exception as e:
        return f"Error: {str(e)}"
    
@app.route('/download_trained_model')
def download_trained_model():
    model_name = request.args.get('model_name', '')  # Get the model name from the query parameters
    if not model_name.endswith('.h5'):
        model_name += '.h5'  # Ensure the model name has the .h5 extension

    model_path = os.path.join(r"C://Users//Admin//Downloads//webappml//", model_name)  # Set the correct directory for your models
    return send_file(model_path, as_attachment=True)




# Add a new route to handle setting deploy file name
@app.route('/set_deploy_filename', methods=['POST'])
def set_deploy_filename():
    global deploy_filename
    deploy_filename = request.form['deploy_filename']
    return f'Deploy file name set to: {deploy_filename}'



@app.route('/run_app1')
def run_app1():
    try:
        subprocess.run(['python', 'app1.py'], check=True)
        return 'app1.py executed successfully!'
    except subprocess.CalledProcessError as e:
        return f'Error executing app1.py: {e}', 500

@app.route('/add_labels', methods=['POST'])
def add_labels():
    global labels_list
    labels_input = request.form['labels']
    new_labels = [label.strip() for label in labels_input.split(',') if label.strip()]
    labels_list.extend(new_labels)

    # Print the labels list to the console
    print("Labels List:", labels_list)

    return 'Labels added successfully!'

@app.route('/get_labels', methods=['GET'])
def get_labels():
    global labels_list
    return jsonify(labels_list)

@app.route('/clear_labels', methods=['GET'])
def clear_labels():
    global labels_list
    labels_list.clear()
    return 'Labels cleared successfully!'

@app.route('/set_deploy_target', methods=['POST'])
def set_deploy_target():
    global deploy_target
    deploy_target = request.form['deploy_target']
    return f'Deploy target set to: {deploy_target}'






@app.route('/upload_model', methods=['POST'])
def upload_model():
    global uploaded_model_path

    model_file = request.files['model']

    if model_file:
        model_filename = secure_filename(model_file.filename)
        project_path = os.getcwd()  # Get the current project directory
        model_file_path = os.path.join(project_path, model_filename)
        model_file.save(model_file_path)
        model_file_path2 = os.path.join(r"C://Users//Admin//Downloads//webappml//models", model_filename)
        model_file.save(model_file_path2)

        # Store the path in the global variable
        uploaded_model_path = model_file_path

        # Print the path to the terminal
        print("Uploaded Model Path:", uploaded_model_path)
         #Print the path to the terminal
        print(labels_list)
        print(deploy_filename)
        print(deploy_target)

        return 'Model file uploaded successfully!'
    else:
        return 'No model file selected.', 400

# Add a new route to handle model deployment, profiling, and file download
@app.route('/deploy_and_download', methods=['GET'])
def deploy_and_download():
    global uploaded_model_path, deploy_target,labels_list,deploy_filename

    # Load the model
    loaded_model = tf.keras.models.load_model(uploaded_model_path)

    # Set model information, such as your list of labels
    model_output_type = ei.model.output_type.Classification(labels=labels_list)

    # Set model input type
    model_input_type = ei.model.input_type.OtherInput()

    # Estimate the RAM, ROM, and inference time for our model on the target hardware family
    try:
        profile = ei.model.profile(model=loaded_model, device=deploy_target)
        print(profile.summary())
    except Exception as e:
        print(f"Could not profile: {e}")

    # Create C++ library with trained model
    deploy_bytes = None
    try:
        deploy_bytes = ei.model.deploy(model=loaded_model, model_output_type=model_output_type,
                                       model_input_type=model_input_type, deploy_target="arduino")
    except Exception as e:
        print(f"Could not deploy: {e}")

    # Write the downloaded raw bytes to a temporary file
    if deploy_bytes:
        temp_deploy_filename = deploy_filename
        with open(temp_deploy_filename, 'wb') as f:
            f.write(deploy_bytes.getvalue())

        return send_file(temp_deploy_filename, as_attachment=True)
    else:
        return "Model deployment failed."
    


@app.route('/update_model', methods=['POST'])
def update_model():
    data = request.json

    # Update the train.py file with the new model architecture and training parameters
    with open('train.py', 'w') as file:
        file.write(f"""
from tensorflow.keras import layers, models, optimizers
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
                   
# Function to load images and labels from folders
def load_data(folder_path):
    images = []
    labels = []
    
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                images.append(img)
                labels.append(label)
                
    return np.array(images), np.array(labels)


# Load data from folders
data_path = r"C://Users//Admin//Downloads//webappml//data"
images, labels = load_data(data_path)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the ImageDataGenerator on the training data
datagen.fit(X_train)

                   
model = models.Sequential()
model.add(layers.Conv2D({data['conv1_filters']}, (3, 3), activation='{data['activation_function']}', input_shape=({data['input_shape']}, {data['input_shape']}, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D({data['conv2_filters']}, (3, 3), activation='{data['activation_function']}'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D({data['conv3_filters']}, (3, 3), activation='{data['activation_function']}'))
model.add(layers.Flatten())
model.add(layers.Dense({data['dense_units']}, activation='{data['activation_function']}'))
model.add(layers.Dense(len(set(labels)), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with augmented data
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs={data['epochs']}, validation_data=(X_test, y_test))

# Save the model
model.save("{data['model_name']}.h5")

print("Model trained with data augmentation and saved successfully.")
    """)

    return 'Model architecture updated successfully!'




