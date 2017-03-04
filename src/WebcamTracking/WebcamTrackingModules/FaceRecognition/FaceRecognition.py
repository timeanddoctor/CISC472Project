import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
import cv2 as cv2

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

  def getVtkImageDataAsOpenCVMat(self, volumeNodeName):
    cameraVolume = slicer.util.getNode(volumeNodeName)
    image = cameraVolume.GetImageData()
    shape = list(cameraVolume.GetImageData().GetDimensions())
    shape.reverse()
    components = image.GetNumberOfScalarComponents()
    if components > 1:
      shape.append(components)
      shape.remove(1)
    imageMat = vtk.util.numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(shape)
    imageMat = imageMat[::-1, ::-1, ::-1]
    imageMat = imageMat.copy()

    return imageMat


  def getOpenCVMatAsVtkImageData(self, imageMat):
    imageMat = np.rot90(imageMat,3)
    imageMat = np.flipud(imageMat)
    destinationArray = vtk.util.numpy_support.numpy_to_vtk(imageMat.transpose(2, 1, 0).ravel(), deep = True)
    destinationImageData = vtk.vtkImageData()    
    destinationImageData.SetDimensions(imageMat.shape)
    destinationImageData.GetPointData().SetScalars(destinationArray)

    return destinationImageData


  def createWebcamPlusConnector(self):
    webcamConnectorNode = slicer.util.getNode('WebcamPlusConnector')
    if not webcamConnectorNode:
      webcamConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
      webcamConnectorNode.SetLogErrorIfServerConnectionFailed(False)
      webcamConnectorNode.SetName('WebcamPlusConnector')
      slicer.mrmlScene.AddNode(webcamConnectorNode)
      webcamConnectorNode.SetTypeClient('localhost', 18944)
      logging.debug('Webcam PlusConnector Created')
    webcamConnectorNode.Start()


  def setupViewerImage(self):
    self.destinationImageVolume = slicer.util.getNode('FaceRecognition_Image')
    if self.destinationImageVolume == None:
      self.destinationImageVolume = slicer.vtkMRMLScalarVolumeNode()
      self.destinationImageVolume.SetName('FaceRecognition_Image')
      self.destinationImageVolume.SetSpacing(0.2, 0.2, 0.2)
      slicer.mrmlScene.AddNode(self.destinationImageVolume)


  def doImageStuff(self, caller, eventid):
    frame = self.getVtkImageDataAsOpenCVMat('Image_Reference')

    # Convert to grayscale image.
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = self.faceCascade.detectMultiScale(
      gray,
      scaleFactor = 1.1,
      minNeighbors = 5,
      minSize = (50, 50)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    destinationImageData = self.getOpenCVMatAsVtkImageData(frame)
    self.destinationImageVolume.SetAndObserveImageData(destinationImageData)


  def run(self):
    """
    Run the actual algorithm
    """
    # Create PLUS Connector
    self.createWebcamPlusConnector()

    # Create the image to be filled on updates.
    self.setupViewerImage()

    self.faceCascade = cv2.CascadeClassifier()
    #faceCascade.load('C:\\Users\\Zac Baum\\Documents\\GitHub\\CISC472Project\\src\\WebcamTracking\\WebcamTrackingModules\\FaceRecognition\\haarcascade_frontalface_default.xml')
    self.faceCascade.load('C:\\Users\\baum\\Documents\\GitHub\\CISC472Project\\src\\WebcamTracking\\WebcamTrackingModules\\FaceRecognition\\haarcascade_frontalface_default.xml')

    self.webcamImageVolume = slicer.util.getNode('Image_Reference')
    self.imageDataModifiedObserver = self.webcamImageVolume.AddObserver(slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent, self.doImageStuff)

    # Set the destination image volume to the red slice background.
    redWidget = slicer.app.layoutManager().sliceWidget('Red')
    redWidget.setSliceOrientation('Reformat')
    redWidget.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.destinationImageVolume.GetID())
    redWidget.sliceLogic().FitSliceToAll()


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
