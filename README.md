# Capstone-Occluded-Object-Detection

This is the GitHub for Capstone Team 5, 3D Occluded Object Detection.

Our system is broken into 4 subsystems: 2D Camera,        3D Camera,     Object Detection/Localization, and Error Detection
The owners of the subsystems are:       Hannah Hillhouse, Dongwon Jeong, Evan Kolin,                    and Samiha Elahi

The Problem:
Need to classify and localize parts in a crowded bin to fulfill a request/order for a list of parts.

The Solution:
Use a 3D camera over the crowded bin, which has access to a trained AI model capable of classifying and localizing parts located within.
Once parts have been localized, they are moved to a seperate empty bin, which then moves on to the 2D camera subsytem.
The 2D camera subsystem also classifies and localizes it's contents so that error detection can check that the correct pieces were chosen.


NOTABLE FILES:
Copy_of_YOLOv5_Custom_Training.ipynb: Training code for YOLOv5
Final_Model.ipynb: Final Tensorflow Model
Final_Version_3D.py: Final code for 3D subsystem
best.pt: Final YOLO Model (whats used in systems)
final_code_2D.py: Final code for 2D subsystem
plots_3d.py: modified YOLO plots file for use in 3D subsystem

DATASET:
GitHub would not allow us to upload our dataset due to size contraints. 
Instead we have stored it on Google Drive, Here is a link to our dataset*
https://drive.google.com/drive/folders/155CM4diYdXsmYyHCA9q1sR526hG_2P91?usp=sharing

YOLO CODE:
Again, we were limited by size constraints.
Here is the files used for YOLOv5 training, and the dependencies for using
YOLO inside of our own code.*
https://drive.google.com/drive/folders/1fDS4crwuZUgS9tC-gKbghGZBG4KLWQNW?usp=sharing

* Must be within TAMU Network to view links
