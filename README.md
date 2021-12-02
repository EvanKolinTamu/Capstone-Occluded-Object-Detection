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
