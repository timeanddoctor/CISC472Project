import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np

#
# FaceRecognition
#

class FaceRecognition(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "FaceRecognition" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["Zachary Baum (PerkLab)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
    Scripted module to use webcam for facial recognition.
    """
    self.parent.acknowledgementText = """
    This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
    and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# FaceRecognitionWidget
#

class FaceRecognitionWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Go!")
    self.applyButton.enabled = True
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = True

  def onApplyButton(self):
    logic = FaceRecognitionLogic()
    logic.run()

#
# FaceRecognitionLogic
#

class FaceRecognitionLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """


  def getVtkImageDataAsOpenCVMat(self):
    cameraVolume = slicer.util.getNode('Image_Reference')
    
    image = vtk.vtkImageData()
    image = cameraVolume.GetImageData()

    caster = vtk.vtkImageCast()
    caster.SetInputData(image)
    caster.SetOutputScalarTypeToUnsignedChar()
    caster.Update()

    dims = image.GetDimensions()
    matImage = np.zeros((dims[0], dims[1], dims[2]), dtype = 'uint8')

    extractVOI = vtk.vtkExtractVOI()
    extractVOI.SetInputData(caster.GetOutput())
    extractVOI.SetVOI(0, dims[0], 0, dims[1], 0, dims[2])
    extractVOI.Update()

    for i in range(0, dims[0]):
      for j in range(0, dims[1]):
        out = extractVOI.GetOutput()
        matImage[(i, j)] = out.GetScalarComponentAsDouble(i, j, 0, 1)

    return matImage

  def run(self):
    """
    Run the actual algorithm
    """
    import cv2

    faceCascade = cv2.CascadeClassifier()
    faceCascade.load('C:\\Users\\baum\\Documents\\GitHub\\CISC472Project\\src\\WebcamTracking\\WebcamTrackingModules\\FaceRecognition\\haarcascade_frontalface_default.xml')
    
    if faceCascade.empty():
      logging.debug('Could not load cascade file!')
      return 0

    #capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        frame = self.getVtkImageDataAsOpenCVMat()

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
          frame,
          scaleFactor=1.1,
          minNeighbors=5,
          minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
          cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    # When everything is done, release the capture
    #capture.release()
    cv2.destroyAllWindows()


class FaceRecognitionTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_FaceRecognition1()

  def test_FaceRecognition1(self):
    return 1