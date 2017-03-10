# CISC472Project
CISC 472 Project - Webcam Object Tracking for 3D Slicer

## Goal:
This project aims to bring colour object tracking via webcam into 3DSlicer as a standalone module. The first integation this project will be used for is the Central Line Tutor, but will in the long term become an aspect of SlicerIGT.

## Installation:
Clone or Fork this repository locally, add the directory ```src``` to your list of modules in 3DSlicer so that it will be loaded when 3DSlicer starts up.

## Project Planner:
Exact content can be found in the PerkTutorPrivate Assembla space, below are high level overviews.

### Closed Tickets:

- Stream Webcam through PLUS into 3DSlicer
- Combined existing PLUS configurations for EM trackinng with Webcam via two instances of a PLUS server
- Use OpenCV to recognize known objects by pre-defined colour
- Be able to recognize and determine the shape of given objects (square, or rectangular)

### TODO:

- Create a method for object selection natively in the module

## Helpful Links:
[QT](http://doc.qt.io/qt-4.8/classes.html) - QT GUI code documentation.

[VTK](http://www.vtk.org/doc/release/6.2/html/classes.html) - Visualization Toolkit documentation.

[MRML / Slicer](https://www.slicer.org/doc/html/classes.html) - Slicer Documentation.

[OpenCV](http://docs.opencv.org/2.4/) - OpenCV Documentation
