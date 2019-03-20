# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(832, 610)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.createModelBtn = QtWidgets.QPushButton(self.centralwidget)
        self.createModelBtn.setGeometry(QtCore.QRect(550, 270, 93, 28))
        self.createModelBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.createModelBtn.setObjectName("createModelBtn")
        self.openModelBtn = QtWidgets.QPushButton(self.centralwidget)
        self.openModelBtn.setGeometry(QtCore.QRect(550, 420, 93, 28))
        self.openModelBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.openModelBtn.setCheckable(False)
        self.openModelBtn.setFlat(False)
        self.openModelBtn.setObjectName("openModelBtn")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(660, 270, 151, 23))
        self.progressBar.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(550, 30, 261, 221))
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.imageLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageLabel.setGeometry(QtCore.QRect(150, 20, 28, 28))
        self.imageLabel.setObjectName("imageLabel")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setEnabled(True)
        self.label_2.setGeometry(QtCore.QRect(550, 10, 101, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 30, 121, 16))
        self.label_3.setObjectName("label_3")
        self.stopModelCreationBtn = QtWidgets.QPushButton(self.centralwidget)
        self.stopModelCreationBtn.setEnabled(False)
        self.stopModelCreationBtn.setGeometry(QtCore.QRect(550, 310, 93, 28))
        self.stopModelCreationBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.stopModelCreationBtn.setObjectName("stopModelCreationBtn")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(550, 460, 101, 31))
        self.label.setObjectName("label")
        self.currentModel = QtWidgets.QLabel(self.centralwidget)
        self.currentModel.setGeometry(QtCore.QRect(660, 460, 161, 31))
        self.currentModel.setObjectName("currentModel")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(126, 90, 101, 31))
        self.label_5.setObjectName("label_5")
        self.predictedValue = QtWidgets.QLabel(self.centralwidget)
        self.predictedValue.setGeometry(QtCore.QRect(230, 90, 51, 31))
        self.predictedValue.setObjectName("predictedValue")
        self.predictValueBtn = QtWidgets.QPushButton(self.centralwidget)
        self.predictValueBtn.setGeometry(QtCore.QRect(20, 90, 93, 28))
        self.predictValueBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.predictValueBtn.setObjectName("predictValueBtn")
        self.plotBtn = QtWidgets.QPushButton(self.centralwidget)
        self.plotBtn.setGeometry(QtCore.QRect(190, 500, 121, 28))
        self.plotBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.plotBtn.setObjectName("plotBtn")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 150, 501, 321))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.plotLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.plotLayout.setContentsMargins(0, 0, 0, 0)
        self.plotLayout.setObjectName("plotLayout")
        self.roadsRadioBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.roadsRadioBtn.setGeometry(QtCore.QRect(560, 350, 111, 20))
        self.roadsRadioBtn.setObjectName("roadsRadioBtn")
        self.digitsRadioBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.digitsRadioBtn.setGeometry(QtCore.QRect(560, 380, 95, 20))
        self.digitsRadioBtn.setChecked(True)
        self.digitsRadioBtn.setObjectName("digitsRadioBtn")
        self.animalsRadioBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.animalsRadioBtn.setGeometry(QtCore.QRect(690, 350, 95, 20))
        self.animalsRadioBtn.setObjectName("animalsRadioBtn")
        self.useConsoleCheckBox = QtWidgets.QCheckBox(self.centralwidget)
        self.useConsoleCheckBox.setGeometry(QtCore.QRect(610, 10, 101, 20))
        self.useConsoleCheckBox.setChecked(False)
        self.useConsoleCheckBox.setTristate(False)
        self.useConsoleCheckBox.setObjectName("useConsoleCheckBox")
        self.aerialImagesRadioBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.aerialImagesRadioBtn.setGeometry(QtCore.QRect(690, 380, 111, 20))
        self.aerialImagesRadioBtn.setObjectName("aerialImagesRadioBtn")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 832, 26))
        self.menubar.setObjectName("menubar")
        self.fileMenu = QtWidgets.QMenu(self.menubar)
        self.fileMenu.setObjectName("fileMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.exitAct = QtWidgets.QAction(MainWindow)
        self.exitAct.setObjectName("exitAct")
        self.openAct = QtWidgets.QAction(MainWindow)
        self.openAct.setShortcutVisibleInContextMenu(True)
        self.openAct.setObjectName("openAct")
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)
        self.menubar.addAction(self.fileMenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.createModelBtn.setText(_translate("MainWindow", "Create Model"))
        self.openModelBtn.setText(_translate("MainWindow", "Open Model"))
        self.imageLabel.setText(_translate("MainWindow", "Image"))
        self.label_2.setText(_translate("MainWindow", "Console"))
        self.label_3.setText(_translate("MainWindow", "Image to predict"))
        self.stopModelCreationBtn.setText(_translate("MainWindow", "Stop Creation"))
        self.label.setText(_translate("MainWindow", "Current Model -> "))
        self.currentModel.setText(_translate("MainWindow", "None"))
        self.label_5.setText(_translate("MainWindow", "Predicted value:"))
        self.predictedValue.setText(_translate("MainWindow", "None"))
        self.predictValueBtn.setText(_translate("MainWindow", "Predict value"))
        self.plotBtn.setText(_translate("MainWindow", "Plot"))
        self.roadsRadioBtn.setText(_translate("MainWindow", "Roads"))
        self.digitsRadioBtn.setText(_translate("MainWindow", "Digits"))
        self.animalsRadioBtn.setText(_translate("MainWindow", "Animals"))
        self.useConsoleCheckBox.setText(_translate("MainWindow", "Use console"))
        self.aerialImagesRadioBtn.setText(_translate("MainWindow", "Aerial Images"))
        self.fileMenu.setTitle(_translate("MainWindow", "File"))
        self.exitAct.setText(_translate("MainWindow", "Exit"))
        self.exitAct.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.openAct.setText(_translate("MainWindow", "Open"))
        self.openAct.setShortcut(_translate("MainWindow", "Ctrl+O"))


