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

    return imageMat


  def getOpenCVMatAsVtkImageData(self, imageMat):
    imageMat = np.rot90(imageMat, 1)
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


  def onWebcamImageModified(self, caller, eventid):
    
    import cv2
    
    # Get the vtkImageData as an np.array.
    imData = self.getVtkImageDataAsOpenCVMat('Image_Reference')
    
    # Define the colour boundaries of the object you want to track.
    boundaries = [([0, 35, 25], [50, 85, 75]),
                  ([108, 187, 108], [158, 237, 250]),
                  ([3, 80, 80], [43, 130, 130]),]

    # Go through each of the boundaries defined and combine the binary images with the original.
    for (lower, upper) in boundaries:
      lower = np.array(lower, dtype = 'uint8')
      upper = np.array(upper, dtype = 'uint8')

      mask = cv2.inRange(imData, lower, upper)
      output = cv2.bitwise_and(imData, imData, mask = mask)

    # Make everything monochrome and threshold
    imgray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the contours and draw them out to the to the original image.
    # The first contour fills the generated lines, second enhances the edges of the contour.
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imData, contours, -1, (0, 255, 0), thickness = -1, maxLevel = 2)
    cv2.drawContours(imData, contours, -1, (0, 255, 0), thickness = 2, maxLevel = 2)


  def run(self):
    import cv2

    self.createWebcamPlusConnector()

    self.webcamImageVolume = slicer.util.getNode('Image_Reference')
    self.imageDataModifiedObserver = self.webcamImageVolume.AddObserver(slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent, self.onWebcamImageModified)

    redWidget = slicer.app.layoutManager().sliceWidget('Red')
    redWidget.setSliceOrientation('Axial')
    redWidget.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.webcamImageVolume.GetID())
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
