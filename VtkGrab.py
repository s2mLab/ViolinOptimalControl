import sys

from PyQt5.QtWidgets import QVBoxLayout,QPushButton, QWidget, QApplication
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def print_screen_callback(w):
    w.grab().save("non_working_print_screen.png")


# Create a small interface that show a cylinder and allow to print it to a PNG file
app = QApplication(sys.argv)
w = QWidget()
w.show()
lay = QVBoxLayout()
w.setLayout(lay)

# Create and populate the vtk widget
ren = vtk.vtkRenderer()
w_vtk = QVTKRenderWindowInteractor()
lay.addWidget(w_vtk)
w_vtk.Initialize()
w_vtk.Start()
w_vtk.GetRenderWindow().AddRenderer(ren)
cylinder = vtk.vtkCylinderSource()
cylinder.SetResolution(8)
cylinderMapper = vtk.vtkPolyDataMapper()
cylinderMapper.SetInputConnection(cylinder.GetOutputPort())
cylinderActor = vtk.vtkActor()
cylinderActor.SetMapper(cylinderMapper)
ren.AddActor(cylinderActor)

# Add print screen button
but = QPushButton("Print")
lay.addWidget(but)
but.released.connect(lambda: print_screen_callback(w))

# Run the application
app.exec()
