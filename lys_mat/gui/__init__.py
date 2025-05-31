
from lys.Qt import QtCore, QtWidgets
from lys.widgets import SidebarWidget
from lys.glb import addSidebarWidget

from .viewer import MoleculeViewer
from .editor import MoleculeEditor


def editViewer(view):
    _edit.setViewer(view)


class _MoleculeEditorBar(SidebarWidget):
    def __init__(self):
        super().__init__("Molecule")
        self._view = None
        self._widget = None
        self.__inilayout()
        self.__setWidget()

    def __inilayout(self):
        self._layout = QtWidgets.QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

    def setViewer(self, viewer):
        if self._view is not None:
            self._view.closed.disconnect(self.__closed)
        self._view = viewer
        self._view.closed.connect(self.__closed)
        self.__setWidget(MoleculeEditor(viewer))
        self.show()

    def __closed(self):
        self._view = None
        self.__setWidget()
        self.show(False)

    def __setWidget(self, wid=None):
        if wid is None:
            wid = QtWidgets.QWidget()
        if self._widget is not None:
            self._layout.removeWidget(self._widget)
            self._widget.deleteLater()
        self._widget = wid
        self._layout.insertWidget(0, self._widget)

_edit = _MoleculeEditorBar()
addSidebarWidget(_edit)
