# Foot Ulcer and Gangrene Detection

A Flask web app that classifies foot photos as **Normal**, **Foot Ulcer (DFU)** or **Gangrene** using a deep learning model (Keras/TensorFlow), with separate Patient, Doctor and Admin portals.

> Academic project. Not a certified medical device. It does not replace professional diagnosis.

## Features

- **Patient portal:** register, send a foot photo, choose a doctor, view results and download the PDF report
- **Admin portal:** see every case (patient-sent and doctor-created) in one dashboard, forward cases to doctors, manage doctors and users, reset a patient's access
- **Doctor portal:** review forwarded cases, run the model, give your own diagnosis and recommendation, send the result to the patient
- **Doctor New Scan:** the doctor can scan a new patient directly by entering the patient's details, email and photo
- **Patient ID activation:** a patient added by a doctor gets a Patient ID by email, then sets their own password on the Activate page
- **PDF report by email:** sent to the patient when the doctor sends the result, with resend buttons for the Patient ID and the report

## Workflow

1. Patient sends a photo (or the doctor scans a new patient directly)
2. Admin forwards patient-sent cases to a doctor
3. Doctor analyzes the photo, adds a diagnosis and recommendation, and sends the result
4. The patient sees the result in the dashboard and receives the PDF by email

## Setup

```bash
python -m venv venv
source venv/Scripts/activate        # Git Bash on Windows
pip install -r requirements.txt
```

Put the trained model in the `model/` folder (see `train_model.py` to train your own).

Create the first admin account with `make_admin.py` (check the script for the details it uses).

## Run

```bash
python app.py
```

The camera needs HTTPS or localhost. To run with HTTPS, put `cert.pem` and `key.pem` in the project folder and start the app with `HTTPS=1`.

## Email settings (optional)

Emails need an SMTP account. For Gmail, create an **App Password** and set these in the terminal before starting the app:

```bash
export SMTP_USER="your_sender@gmail.com"
export SMTP_PASS="your_app_password"
# optional: SMTP_HOST (default smtp.gmail.com), SMTP_PORT (default 587)
```

Without these, the app still works. Doctors are shown the Patient ID on screen and the report can be downloaded from the case page.

Never commit passwords, `key.pem`, `cert.pem`, the database or patient images.

## Project structure

- `app.py` Flask routes, models and report generation
- `templates/` HTML pages
- `static/` CSS and uploaded assets
- `train_model.py`, `data_preprocessing.py` model training and preprocessing
- `model/` trained model
