import sys
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def create_sample_volume(nx=80, ny=80, nz=80):
    """
    生成一份 3D 标量场数据：
    两个高斯球叠加，返回 vtkImageData
    """
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    z = np.linspace(-1.0, 1.0, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    g1 = np.exp(-((X + 0.35) ** 2 + Y**2 + Z**2) / 0.08)
    g2 = 0.8 * np.exp(-((X - 0.25) ** 2 + (Y - 0.2) ** 2 + (Z + 0.15) ** 2) / 0.05)
    data = (g1 + g2).astype(np.float32)

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.SetOrigin(0.0, 0.0, 0.0)

    # VTK 需要列主序/Fortran 风格展平
    flat_data = np.ascontiguousarray(data.transpose(2, 1, 0)).ravel()

    vtk_array = vtk.vtkFloatArray()
    vtk_array.SetNumberOfValues(flat_data.size)
    for i, v in enumerate(flat_data):
        vtk_array.SetValue(i, float(v))

    image.GetPointData().SetScalars(vtk_array)
    return image, float(data.min()), float(data.max())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 + VTK 三维数据显示原型")
        self.resize(1200, 800)

        self.image_data, self.data_min, self.data_max = create_sample_volume()

        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)

        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(260)

        self.value_label = QLabel("等值面阈值: 0.35")
        self.value_label.setAlignment(Qt.AlignCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setValue(35)
        self.slider.valueChanged.connect(self.on_slider_changed)

        info = QLabel(
            "数据说明：\n"
            "- 这里是程序生成的 3D 标量场\n"
            "- 使用 Marching Cubes 提取等值面\n"
            "- 拖动滑块查看不同阈值下的形状"
        )
        info.setWordWrap(True)

        control_layout.addWidget(self.value_label)
        control_layout.addWidget(self.slider)
        control_layout.addSpacing(20)
        control_layout.addWidget(info)
        control_layout.addStretch()

        # 右侧 VTK 视图
        self.vtk_widget = QVTKRenderWindowInteractor(central)
        layout.addWidget(control_panel)
        layout.addWidget(self.vtk_widget, 1)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.12, 0.14, 0.18)

        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        self.interactor = self.render_window.GetInteractor()

        self.setup_pipeline()
        self.add_axes()
        self.renderer.ResetCamera()

    def setup_pipeline(self):
        # 等值面提取
        self.mc = vtk.vtkMarchingCubes()
        self.mc.SetInputData(self.image_data)
        self.mc.ComputeNormalsOn()
        self.mc.ComputeGradientsOn()
        self.mc.SetValue(0, 0.35)

        # 平滑一下显示效果
        self.smoother = vtk.vtkSmoothPolyDataFilter()
        self.smoother.SetInputConnection(self.mc.GetOutputPort())
        self.smoother.SetNumberOfIterations(20)
        self.smoother.SetRelaxationFactor(0.1)
        self.smoother.FeatureEdgeSmoothingOff()
        self.smoother.BoundarySmoothingOn()

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.smoother.GetOutputPort())
        self.mapper.ScalarVisibilityOff()

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetColor(0.3, 0.8, 1.0)
        self.actor.GetProperty().SetSpecular(0.3)
        self.actor.GetProperty().SetSpecularPower(20)

        self.renderer.AddActor(self.actor)

    def add_axes(self):
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(15, 15, 15)

        self.marker = vtk.vtkOrientationMarkerWidget()
        self.marker.SetOrientationMarker(axes)
        self.marker.SetInteractor(self.interactor)
        self.marker.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.marker.SetEnabled(1)
        self.marker.InteractiveOff()

    def on_slider_changed(self, value):
        iso = value / 100.0
        self.value_label.setText(f"等值面阈值: {iso:.2f}")
        self.mc.SetValue(0, iso)
        self.render_window.Render()

    def showEvent(self, event):
        super().showEvent(event)
        self.interactor.Initialize()
        self.interactor.Start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())