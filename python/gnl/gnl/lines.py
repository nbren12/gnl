# -*- coding: utf-8 -*-
"""
This example demonstrates many of the 2D plotting capabilities
in pyqtgraph. All of the plots may be panned/scaled by dragging with
the left/right mouse buttons. Right click on any plot to show a context menu.
"""


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

import gnl
import pandas as pd


class MyGraphicsWindow(pg.GraphicsWindow):
    plotpane = 0


    def __init__(self, phi):
        super(MyGraphicsWindow, self).__init__()

        self.phi  = phi
        xlim = (10000, 11000)
        nphi = 10
        ncol = 2
        nrow = nphi / ncol
        self.nphi = nphi

        # Plot
        self.lines = []
        self.plots = []
        for i in range(nphi):
            p = self.addPlot(title=i)
            line = p.plot([],[])
            p.setXRange(*xlim)

            self.lines.append(line)
            self.plots.append(p)

            if i%2 == 1:
                self.nextRow()
            if i > 0:
                p.setXLink(self.plots[0])
        self.renderPlots()

    def renderPlots(self):
        phi = self.phi
        for i in range(self.nphi):
            col0 = self.plotpane*self.nphi
            self.lines[i].setData(y=phi.iloc[:,col0 + i].values, x=phi.index)
            self.plots[i].setTitle(phi.columns[col0+i])

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Right:
            self.plotpane +=1
            self.renderPlots()
        elif ev.key() == QtCore.Qt.Key_Left:
            self.plotpane -=1
            self.renderPlots()


def test_main():
    import sys
    arr = np.genfromtxt(sys.argv[1])
    phi = pd.DataFrame(arr[:,1:], index=np.int64( arr[:,0] ))
    win = MyGraphicsWindow(phi)
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    test_main()
