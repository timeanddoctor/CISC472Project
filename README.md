# CISC472Project
CISC 472 Project - Webcam Object Tracking for 3D Slicer

## Goal:
This project aims to bring colour object tracking via webcam into 3DSlicer as a standalone module. The first integation this project will be used for is the Central Line Tutor, but will in the long term become an aspect of SlicerIGT.

## Installation:
Clone or Fork this repository locally, add the directory ```src``` to your list of modules in 3DSlicer so that it will be loaded when 3DSlicer starts up.

## Project Planner:
Exact content can be found in the PerkTutorPrivate Assembla space, below are high level overviews.

### Closed Tickets:

- [Stream Webcam through PLUS into 3DSlicer](https://github.com/zacbaum/CISC472Project/commit/d3077fc318a2ee431f9a0da6402a2aed831ff827)
- Combined existing PLUS configurations for EM trackinng with Webcam via two instances of a PLUS server (Commit found on PerkTutorPrivate)
- [Use OpenCV to recognize known objects by pre-defined colour](https://github.com/zacbaum/CISC472Project/commit/5bafaf0bf0d0cf237690c5678651e66e32ac91b8)
- [Be able to recognize and determine the shape of given objects (square, or rectangular)](https://github.com/zacbaum/CISC472Project/commit/4f02526996466cd7bc14f68b15b160046000324d)
- [Ability to track multiple objects at once, list them in table form](https://github.com/zacbaum/CISC472Project/commit/0eb4435bf3a09d55d5eefed5bbc77aedbc2fb661)
- [Option to delete objects from the list of tracked tools in the scene](https://github.com/zacbaum/CISC472Project/commit/8aed290e9bc956b75a6ba4c52028b3d9e2388038)

### TODO:

- Create a method for object selection natively in the module (WIP - [Currently averaging R G B values independently and creating a working racnge around them(https://github.com/zacbaum/CISC472Project/blob/8aed290e9bc956b75a6ba4c52028b3d9e2388038/src/WebcamTracking/WebcamTrackingModules/ColourObjectTracking/ColourObjectTracker.py#L251-L280))
- Add checkbox to show contours on object to verify color selection

## Helpful Links:
[QT](http://doc.qt.io/qt-4.8/classes.html) - QT GUI code documentation.

[VTK](http://www.vtk.org/doc/release/6.2/html/classes.html) - Visualization Toolkit documentation.

[MRML / Slicer](https://www.slicer.org/doc/html/classes.html) - Slicer Documentation.

[OpenCV](http://docs.opencv.org/2.4/) - OpenCV Documentation
