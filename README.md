# Contactless_Vital_Sign_Measure
This repo is for measuring the heart rate and respiration rate using the webcam. Working on SpO2 oxygen level and will try for blood pressure, body temperature too.

### Prerequisites
Ensure the computer you are using to run this project has a webcam. 

This website now runs on a local host so needs these installed:
```
pip install -r requirements.txt
```
Running with Python 3.8

Can be deployed using heroku on a https:// website.

### Usage

Once the project is downloaded. Run the python file:

```
python app.py
```
and then, click on the link of website which pops up:
```
 http://127.0.0.1:5000/ (local server)
```

The webcam light should activate and there should be a window that appears. The program takes a few seconds to detect the pulse and start calculating your heart rate so be patient.


# Implementation

<!--- - In case of plotting graphs, run "graph_plot.py" - For the Eulerian Video Magnification implementation, run "amplify_color.py" --->
  -If want to run individually on GUI, 
  ```
      run "main.py" for Heart Rate, "main-r.py" for Respiration Rate and "main-s.py" for SpO2 Oxygen level. 
  ```
