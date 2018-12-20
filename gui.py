import sys
from PyQt5 import uic, QtCore

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5.QtSql import QSqlDatabase, QSqlTableModel

import main as prog


@pyqtSlot()
def on_click():
    f = open("log.txt", "w")
    f.close()

    lst1 = prog.getCountOfPeople()
    prog.aligning()
    _ = prog.clear_non_trainable()
    lst2 = prog.getCountOfPeople()
    if ((lst2 - lst1) > 0 and lst1 != 0) or ((lst2 - lst1) > 1 and lst1 == 0):
        prog.train()
    persons = prog.do_everything(1)
    window.dataimage.setPixmap(QPixmap("data.jpg").scaled(window.dataimage.width(), window.dataimage.height(), QtCore.Qt.KeepAspectRatio))
    window.faceimage.setPixmap(QPixmap("face.jpg").scaled(window.faceimage.width(), window.faceimage.height(), QtCore.Qt.KeepAspectRatio))
    window.personName.setText(", ".join(persons))

    model = QSqlTableModel()
    model.setTable("personid")
    model.setEditStrategy(QSqlTableModel.OnFieldChange)
    model.select()
    window.tableView.setModel(model)

    window.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = uic.loadUi("view.ui")
    window.dataimage.setPixmap(QPixmap("data.jpg").scaled(window.dataimage.width(), window.dataimage.height(), QtCore.Qt.KeepAspectRatio))
    window.faceimage.setPixmap(QPixmap("face.jpg").scaled(window.faceimage.width(), window.faceimage.height(), QtCore.Qt.KeepAspectRatio))
    window.startButton.clicked.connect(on_click)

    db = QSqlDatabase.addDatabase('QSQLITE')
    db.setDatabaseName('DB')
    db.open()
    model = QSqlTableModel()
    model.setTable("personid")
    model.setEditStrategy(QSqlTableModel.OnFieldChange)
    model.select()
    window.tableView.setModel(model)

    window.show()
    window.update()
    sys.exit(app.exec_())
