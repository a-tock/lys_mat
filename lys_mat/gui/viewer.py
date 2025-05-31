from lys.Qt import QtCore, QtWidgets
from lys.widgets import LysSubWindow

from .widget import MoleculeViewWidget

class MoleculeViewer(LysSubWindow):
    def __init__(self, crystal, **kwargs):
        super().__init__(floating=False, **kwargs)
        self.canvas = MoleculeViewWidget(crystal)
        self.setWidget(self.canvas)
        self.canvas.show()

    def __getattr__(self, key):
        if hasattr(self.canvas, key):
            return getattr(self.canvas, key)
        return super().__getattr__(key)


