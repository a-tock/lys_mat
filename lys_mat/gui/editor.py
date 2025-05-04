from lys.Qt import QtCore, QtWidgets
from lys.widgets import LysSubWindow

#from .Tabs import ViewTab, AtomsTab


class MoleculeEditor(LysSubWindow):
    def __init__(self, parent):
        super().__init__(floating=parent.isFloating())
        self.__parent = parent
        self.__initlayout()
        self.attach(parent)
        self.attachTo()

    def __initlayout(self):
        self.setWindowTitle("Modify molecule/crystal")
        self._tab = QtWidgets.QTabWidget()
        #self._tab.addTab(GeneralTab(self.__parent), "General")
        #self._tab.addTab(ViewTab(self.__parent), "View")
        #self._tab.addTab(QtWidgets.QWidget(), "Data")
        #self._tab.addTab(AtomsTab(self.__parent), "Atoms")
        #self._tab.addTab(QtWidgets.QWidget(), "Bonds")
        #self._tab.addTab(ExportTab(self.__parent), "Export")
        self.setWidget(self._tab)
        self.adjustSize()
        self.updateGeometry()


class GeneralTab(QtWidgets.QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.__viewer = viewer
        self.__initlayout()

    def __initlayout(self):

        self.frame = FrameWidget(self.__viewer.getNumberOfFrames())
        self.frame.valueChanged.connect(self.__viewer.setFrame)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.frame)
        layout.addStretch()
        self.setLayout(layout)


class FrameWidget(QtWidgets.QGroupBox):
    valueChanged = QtCore.pyqtSignal(int)

    def __init__(self, frames):
        super().__init__("Frame control")
        self.__initlayout(frames)

    def __initlayout(self, frames):
        self.value = QtWidgets.QSpinBox()
        self.value.setRange(0, frames - 1)
        self.value.valueChanged.connect(self.valueChanged.emit)
        self.value.valueChanged.connect(self.__onValueChange)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, frames - 1)
        self.slider.valueChanged.connect(self.value.setValue)
        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(QtWidgets.QLabel("Frame Number"))
        h1.addWidget(self.value)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(h1)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def __onValueChange(self, value):
        self.slider.valueChanged.disconnect(self.value.setValue)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self.value.setValue)

    def getFrame(self):
        return self.value.value()

    def setFrame(self, ranges):
        self.value.setValue()


class ExportTab(QtWidgets.QWidget):
    def __init__(self, viewer):
        super().__init__()
        self._viewer = viewer

        self._anim_name = QtWidgets.QLineEdit()
        self._anim_name.setText("animation1")
        self._width = QtWidgets.QSpinBox()
        self._width.setRange(0, 10000)
        self._width.setValue(600)
        self._height = QtWidgets.QSpinBox()
        self._height.setRange(0, 10000)
        self._height.setValue(400)
        b1 = QtWidgets.QPushButton("Export animation", clicked=self.__anim)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Name"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("Width"), 1, 0)
        grid.addWidget(QtWidgets.QLabel("Height"), 1, 2)
        grid.addWidget(self._anim_name, 0, 1)
        grid.addWidget(self._width, 1, 1)
        grid.addWidget(self._height, 1, 3)
        grid.addWidget(b1, 2, 0, 1, 4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)
        grp1 = QtWidgets.QGroupBox("Animation")
        grp1.setLayout(grid)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QPushButton("Ray trace", clicked=self._viewer.ray))
        layout.addWidget(QtWidgets.QPushButton("Export image", clicked=lambda: self._viewer.exportImage("image.png")))
        layout.addWidget(grp1)
        layout.addStretch()
        self.setLayout(layout)

    def __anim(self):
        self._viewer.exportAnimation(self._anim_name.text(), self._width.value(), self._height.value())
        QtWidgets.QMessageBox.information(self, "Information", "Animation is exported to " + self._anim_name.text() + ".mp4")
