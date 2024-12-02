EasyTinyML: An Automated TinyML Model Deployment Platform

Simplify your journey from trained models to microcontroller deployment.

Overview
EasyTinyML is a web-based platform designed to automate the deployment of TinyML models on microcontrollers. Users can upload trained machine learning models as .h5 files, which are automatically converted to a format suitable for TinyML deployment. The platform also supports direct training of object detection models, enabling seamless deployment on MCU targets without the need for manual programming in IDEs like Arduino.

Features
Automated Model Conversion:
Upload .h5 files and let the platform handle the conversion to a TinyML-compatible format for microcontrollers.

Quick Model Training:
Train object detection models directly on the platform, eliminating the need for external tools.

Effortless MCU Deployment:
Bypass complex IDE setups by embedding models onto microcontrollers with minimal effort.

User-Friendly Interface:
Intuitive GUI for selecting target MCUs and managing deployment parameters.

Getting Started
Clone the Repository:

bash
Copy code
git clone https://github.com/username/EasyTinyML.git
cd EasyTinyML
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Platform:

bash
Copy code
python app.py
Access the Web Interface:
Open your browser and navigate to http://localhost:5000.

Workflow
Upload your .h5 model file.
Select your target microcontroller.
The platform converts and prepares the model for deployment.
Download the ready-to-deploy code and flash it to your MCU.
Supported Devices
Arduino Nano 33 BLE Sense
ESP32
Contributing
We welcome contributions! Please fork the repository and submit a pull request.

License
This project is licensed under the MIT License.

