# AI-Powered Real-Time Theft Detection System

## Project Overview

This project focuses on detecting suspicious and theft-related activities in shopping malls using AI techniques applied to CCTV surveillance data. The system analyzes human behavior in real time using pose estimation and temporal modeling.

## Project Status

* YOLOv8 Pose model integrated for human keypoint extraction
* LSTM model trained for temporal behavior analysis
* Real-time inference pipeline implemented
* Backend connected with frontend interface

## System Architecture

The system follows a modular pipeline:

* Video input (CCTV or webcam)
* Pose estimation using YOLOv8
* Feature processing from keypoints
* Temporal analysis using LSTM
* Backend server for inference
* Frontend interface for visualization

## Features

* Real-time theft detection
* Human activity analysis using pose estimation
* Temporal behavior modeling with LSTM
* Web-based interface for monitoring
* Modular and extensible design

## Model Download

Due to GitHub file size limitations, trained models are not included in this repository.

Download them from the following links:

YOLOv8 Pose Model:
(https://drive.google.com/file/d/1fxNgCsW2rGb0wEnFMD7vX8aMpDeAHBct/view?usp=drive_link)

LSTM Model:
(https://drive.google.com/file/d/1Tn0a4RCE4PxBVqrFB4ZCha8I028rJ3om/view?usp=drive_link)

## Model Placement

After downloading, place the model files inside:

backend/models/

Final structure:

backend/
  -inference_server.py
  - models/
    -- yolov8m-pose.pt
    -- best_theft_lstm_model_v6.pth

## Installation

Clone the repository:

git clone https://github.com/Muhammad-Nadeem-229366/AI-Powered-Real-Time-Theft-Detection.git
cd AI-Powered-Real-Time-Theft-Detection

Install dependencies:

pip install -r requirements.txt

## How to Run

Start the backend server:

cd backend
python inference_server.py

Then open the frontend file:

frontend/index.html

## Project Structure

backend/ – Backend server and inference logic
frontend/ – User interface
model_development/ – Training and experimentation notebooks
dataset/ – Dataset (optional or sample)
docs/ – Documentation
diagrams/ – System diagrams
presentations/ – Project presentations

## Objectives

* Analyze CCTV video streams for human activity
* Extract meaningful features such as pose and motion
* Detect suspicious behavior in real time
* Generate alerts and display results

## Design Approach

* Modular architecture for flexibility
* Real-time or near real-time processing
* Designed for scalability and future improvements

## Note

Ensure that model files are placed correctly inside the backend/models/ directory before running the system.
