import math
import time
import itertools

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from lys.Qt import QtWidgets, QtCore, QtGui
from lys_mat import CrystalStructure

from .. import Atom


class _Plotter(QtInteractor):
    mouseReleased = QtCore.pyqtSignal(object)
    mousePressed = QtCore.pyqtSignal(object)
    mouseMoved = QtCore.pyqtSignal(object)
    focused = QtCore.pyqtSignal(object)
    keyPressed = QtCore.pyqtSignal(QtGui.QKeyEvent)
    doubleClicked = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._clicktime = 0

    def enableRendering(self, b):
        self._render = b

    def render(self):
        if self._render:
            return super().render()

    def mouseReleaseEvent(self, event):
        self.mouseReleased.emit(event)
        if time.time() - self._clicktime < 0.3:
            self.doubleClicked.emit()
        self._clicktime = time.time()
        super().mouseReleaseEvent(event)

    def mousePressEvent(self, event):
        self.mousePressed.emit(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.mouseMoved.emit(event)
        super().mouseMoveEvent(event)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.focused.emit(event)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        self.keyPressed.emit(event)

    def _mouseReleased(self, e):
        self.clicked.emit(e)


class MoleculeViewWidget(QtWidgets.QWidget):
    closed = QtCore.pyqtSignal(object)
    
    def __init__(self, crystal, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.__initlayout()
        self.plotter.doubleClicked.connect(self.__editMolecule)
        self.setCrystal(crystal)

        #self.plotter.mouseMoved.connect(self.mouseMoved)
        #self.plotter.mouseReleased.connect(self.mouseReleased)
        #self.plotter.mousePressed.connect(self.mousePressed)
        #self.plotter.focused.connect(self.focused)
        #self.plotter.keyPressed.connect(self.keyPressed)

    def closeEvent(self, event):
        if not event.isAccepted():
            return
        self.closed.emit(event)

    def __initlayout(self):
        self._plotter = _Plotter()
        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addWidget(self._plotter.interactor)
        self.setLayout(vlayout)

    def __editMolecule(self):
        from . import editViewer
        editViewer(self)

    @property
    def plotter(self):
        return self._plotter

    def enableRendering(self, b):
        self.plotter.enableRendering(b)

    def setCrystal(self, crystal):
        """
        Set crystal structure for this widget.

        Args:
            crystal(CrystalStructure of list of CrystalStructure): The crystal structure to be shown.
        """
        if isinstance(crystal, CrystalStructure):
            return self.setCrystal([crystal])
        self.__crys = crystal
        self.__actor = None
        self.__ranges = [(0, 1), (0, 1), (0, 1)]
        self.__atoms = self.__createAtoms(self.__crys, self.__ranges)
        self._update(self.__atoms)

    @property
    def numberOfFrames(self):
        return len(self.__crys)

    def setFrame(self, index):
        """
        Set frame index of the crystal structure to be shown.

        Args:
            index(int): The index.
        """
        if index > len(self.__crys):
            raise RuntimeError("The index out of range.")
        self._update(self.__atoms, index)

    def _update(self, atoms, index=0):
        self.enableRendering(False)
        pts = pv.PolyData([at.position[index] for at in atoms])
        pts["scales"] = [at.size for at in atoms]
        colors = np.array([at.color for at in atoms], dtype=np.uint8)

        sphere = pv.Sphere(radius=1)
        glyphs = pts.glyph(scale='scales', geom=sphere)

        n_glyphs = pts.n_points
        colors_repeated = np.repeat(colors, glyphs.n_points // n_glyphs, axis=0)
        glyphs['rgb'] = colors_repeated

        if self.__actor is None:
            self.__actor = self.plotter.add_mesh(glyphs, scalars="rgb", rgb=True)
        else:
            self.plotter.remove_actor(self.__actor, reset_camera=False)
            self.__actor = self.plotter.add_mesh(glyphs, scalars="rgb", rgb=True)
        self.enableRendering(True)

    def __createAtoms(self, crystals, ranges, eps=1e-5):
        amin, bmin, cmin = math.floor(ranges[0][0]), math.floor(ranges[1][0]), math.floor(ranges[2][0])
        amax, bmax, cmax = math.ceil(ranges[0][1]) + 1, math.ceil(ranges[1][1]) + 1, math.ceil(ranges[2][1]) + 1

        positions = []
        for crys in crystals:
            pos, elems = [], []
            inv = np.linalg.inv(crys.unit)
            for a, b, c in itertools.product(range(amin, amax), range(bmin, bmax), range(cmin, cmax)):
                s = np.array([a,b,c]).dot(crys.unit)
                for at, p in zip(crys.atoms, crys.getAtomicPositions()):
                    r = inv.dot(p + s)
                    if np.array([ranges[i][0]-eps < r[i] < ranges[i][1]+eps for i in range(3)]).all():
                        pos.append(p+s)
                        elems.append(at.Element)
            elements = elems
            positions.append(pos)
        return [AtomView(e, p) for e, p in zip(elements, np.array(positions).transpose(1,0,2))]
        

class AtomView:
    _vdw_rad = [0.53, 0.31, 1.67, 1.12, 0.87, 0.67, 0.56, 0.48, 0.42, 0.38, 1.9, 1.45, 1.18, 1.11, 0.98, 0.88, 0.79, 0.71, 2.43, 1.94, 1.84, 1.76, 1.71, 1.66, 1.61, 1.56, 1.52, 1.49, 1.45, 1.42, 1.36, 1.25, 1.14, 1.03, 0.94, 0.88, 2.65, 2.19, 2.12, 2.06, 1.98, 1.9, 1.83, 1.78, 1.73, 1.69, 1.65, 1.61, 1.56, 1.45, 1.33, 1.23, 1.15, 1.08, 2.98, 2.53, 'no data', 'no data', 2.47, 2.06, 2.05, 2.38, 2.31, 2.33, 2.25, 2.28, 2.27, 2.26, 2.22, 2.22, 2.17, 2.08, 2.0, 1.93, 1.88, 1.85, 1.8, 1.77, 1.74, 1.71, 1.56, 1.54, 1.43, 1.35, 'no data', 1.2, 'no data', 'no data', 'no data', 'no data', 'no data', 'no data', 'no data', 'no data', 'no data', 'no data', 'no data', 'no data', 'no data']

    def __init__(self, element, pos):
        self._elem = element
        self._pos = pos
        self._size = 0.5
        self._color = (255, 255, 0)

    @property
    def position(self):
        return self._pos

    @property
    def size(self):
        return self._size * self._vdw_rad[Atom.getAtomicNumber(self._elem)]

    @property
    def color(self):
        return self._color