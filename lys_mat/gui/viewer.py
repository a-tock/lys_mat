from lys.Qt import QtCore, QtWidgets
from lys.widgets import LysSubWindow

from .widget import MoleculeViewWidget
from .editor import MoleculeEditor

class MoleculeViewer(LysSubWindow):
    def __init__(self, file=None, **kwargs):
        super().__init__(floating=True, **kwargs)
        self.canvas = MoleculeViewWidget()
        self.setWidget(self.canvas)
        self._edit = None
        self.canvas.plotter.doubleClicked.connect(self.__editMolecule)
        self.canvas.show()

    def __getattr__(self, key):
        if hasattr(self.canvas, key):
            return getattr(self.canvas, key)
        return super().__getattr__(key)

    def __editMolecule(self):
        if self._edit is not None:
            self._edit.close()
        else:
            self._edit = MoleculeEditor(self)
            self._edit.adjustSize()
            self._edit.attachTo()
            self._edit.closed.connect(self.__closeEdit)

    def __closeEdit(self):
        self._edit = None