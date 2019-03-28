try:
    import sys
    from datetime import datetime

    # Design
    from .design import Ui_MainWindow

    from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog
    from PyQt5.QtGui import QTextCursor
    from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject, QThread    

    import tensorflow as tf
    import numpy as np

    # Plots
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

    from PIL import Image, ImageQt

    # Importing keras
    from keras import backend as K

    # In our case we're using the GPU which is faster than the CPU
    GPU, CPU = True, False
    num_cores = 8

    if GPU:
        num_GPU = 1
        num_CPU = 1
    if CPU:
        num_CPU = 1
        num_GPU = 0
    # Create a new Tensorflow configuration
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores, 
                            allow_soft_placement=True,
                            device_count = {'CPU' : num_CPU,
                                            'GPU' : num_GPU},
                            gpu_options=tf.GPUOptions(allow_growth=True)
                           )
    # Create a new session with this configuration
    session = tf.Session(config=config)
    # Set the Keras session with the one from Tensorflow
    K.set_session(session)

    # Import CNN models
    from nnmodels import AnimalsModel
    from nnmodels import DigitsModel
    from nnmodels import RoadsModel
    from nnmodels import AerialBuildingsModel
    from nnmodels import AerialRoadsModel
    from nnmodels import KidneysModel
except ImportError as err:
    exit(err)

#tf_session = K.get_session() # this creates a new session since one doesn't exist already.
tf_session = K.get_session()
tf_graph = tf.get_default_graph()

class ThreadCreateModel(QThread):
    """
    TODO: documentation.
    """
    threadDone = pyqtSignal(bool)

    def __init__(self, mw):
        QThread.__init__(self)
        # Main window instance
        self.__mw = mw

        self.threadDone.connect(self.__mw.done)

    def __del__(self):
        self.wait()

    def run(self):
        """
        Main function -> it runs the current thread until its work is done.
        """
        with tf_session.as_default():
            with tf_graph.as_default():
                self.__mw.getModel().learn()
                
                self.threadDone.emit(True)

    def stop(self):
        """
        Stop the current thread.
        """
        self.terminate()
        self.threadDone.emit(False)

class EmittingStream(QObject):
    """
    TODO: documentation.
    """
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        """
        Necessary to avoid this error -> AttributeError: 'EmittingStream' object has no attribute 'flush'
        """
        pass

class MainWindow(QMainWindow):
    """description of class"""

    def __init__(self, x, y, title="Main Window"):
        """
        Class constructor.
        """
        super().__init__()

        # Window size
        self.__x, self.__y, = x, y
        # Title
        self.__tile = title

        # File for MRI loading
        self.__MRIfilename = None
        # File for model loading
        self.__modelFilename = None

        self.__data_to_process = "digits"

        # Initialize the model
        self.__initModel()
        # Initialize the main UI
        self.__initUI()

    def __del__(self):
        """
        Class destructor.
        """
        # Restore sys.stdout
        sys.stdout = sys.__stdout__

    def getModel(self):
        """
        Returns an instance of the current model.
        """
        return self.__model

    def __initModel(self):
        with tf_session.as_default():
            with tf_graph.as_default():
                # Create a variable for neural network model
                self.__model = DigitsModel()
                # Create a variable to prevent the use of an uninitialized model
                self.__isModelLoaded = False
                # Image to predict
                self.__imgToPredict = None

    def __initUI(self):
        """
        Initialization of the main window UI.
        """
        # Create the main UI
        self.__ui = Ui_MainWindow()
        self.__ui.setupUi(self)
        # Create plots
        self.__createPlots()
        # Create connections
        self.__createConnections()
        # Show main window
        self.show()

    def __createPlots(self):
        """
        Creates plots to show model learning performance.
        """
        # A figure instance to plot on
        self.__figure = plt.figure()

        # This is the Canvas Widget that displays the 'figure'
        # It takes the 'figure' instance as a parameter to __init__
        self.__canvas = FigureCanvas(self.__figure)

        # This is the Navigation Widget
        # It takes the Canvas widget and a parent as parameters to __init__
        self.__toolbar = NavigationToolbar(self.__canvas, self)

        self.__ui.plotLayout.addWidget(self.__toolbar)
        self.__ui.plotLayout.addWidget(self.__canvas)

    def __createConnections(self):
        """
        Creates connections between actions/buttons/etc and methods.
        """
        # Actions
        self.__ui.exitAct.triggered.connect(self.close)
        self.__ui.openAct.triggered.connect(self.openFile)
        # Buttons
        self.__ui.createModelBtn.clicked.connect(self.createModel)
        self.__ui.openModelBtn.clicked.connect(self.openModel)
        self.__ui.predictValueBtn.clicked.connect(self.predictValue)
        self.__ui.plotBtn.clicked.connect(self.plot)
        # Radio buttons
        self.__ui.roadsRadioBtn.clicked.connect(lambda: self.setModel("roads"))
        self.__ui.digitsRadioBtn.clicked.connect(lambda: self.setModel("digits"))
        self.__ui.animalsRadioBtn.clicked.connect(lambda: self.setModel("animals"))
        self.__ui.aerialBuildingsRadioBtn.clicked.connect(lambda: self.setModel("aerial_buildings"))
        self.__ui.aerialRoadsRadioBtn.clicked.connect(lambda: self.setModel("aerial_roads"))
        self.__ui.kidneysRadioBtn.clicked.connect(lambda: self.setModel("kidneys"))
        # Console check box
        self.__ui.useConsoleCheckBox.stateChanged.connect(lambda: self.useConsole(self.__ui.useConsoleCheckBox))
 
    def closeEvent(self, event):
        """
        Before closing the main window we ask user agreement.
        """
        reply = QMessageBox.question(self, 
                "Message",
                "Are you sure you want to quit ?",
                QMessageBox.Yes,
                QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    @pyqtSlot()
    def useConsole(self, checkBox):
        """
        Gives two choices to the user:
            - use the EmittingStream class to print in a QTextEdit
            - use the default sys.__stdout__ ouput
        """
        if checkBox.isChecked():
            # Install the custom output stream
            sys.stdout = EmittingStream(textWritten = self.normalOutputWritten) 
        else:
            # Restore sys.stdout
            sys.stdout = sys.__stdout__

    @pyqtSlot(str)
    def normalOutputWritten(self, text):
        """
        Appends text to the QTextEdit.
        """
        # If we're currently generating a new neural network model
        if self.__model.isTraining():
            txtSplit = text.split(" ")
            # Get the epoch and show it in the progress bar
            if txtSplit[0] == "Epoch":
                step        = txtSplit[1].split("/")
                actual_step = step[0]
                max_step    = step[1]
                self.__ui.progressBar.setValue(((int(actual_step)/int(max_step))*100) % 101) # Modulo 101 -> [0;100] and not [0;100[

        cursor = self.__ui.textEdit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.__ui.textEdit.setTextCursor(cursor)
        self.__ui.textEdit.ensureCursorVisible()
    
    @pyqtSlot()
    def openFile(self):
        """
        Opens an existing file.
        """
        # Set options
        options = QFileDialog.Options()
        # Get file path
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "","PNG Files (*.png);;JPG Files (*.jpg;*.jpeg);;TIF Files (*.tif);;HDR Files (*.hdr);;NII Files (*.nii);;All Files (*)", options=options)
        if filename.endswith(tuple([self.__model.FILE_EXTENSIONS])):
            self.__imgToPredict = filename
            self.__model.loadDataToPredict(filename)
        else:
            print("ERROR: wrong file extension.")

    @pyqtSlot()
    def openModel(self):
        """
        Opens an existing model.
        """
        # Set options
        options = QFileDialog.Options()
        # Get file path
        self.__modelFilename, _ = QFileDialog.getOpenFileName(self, "Open Architecture File", "","JSON Files (*.json);;HDF5 Files (*.hdf5);;All Files (*)", options=options)
        # Avoid empty file name
        if len(self.__modelFilename) != 0:
            if self.__modelFilename.endswith(tuple(["json", "hdf5"])):
                # Extract the name
                self.__modelFilename = self.__modelFilename.split(".")[0]
                # Open the model
                self.__model.openModel(self.__modelFilename+".json", self.__modelFilename+".hdf5")
                # Allow the use of the model
                self.__isModelLoaded = True

                try:
                    filename = self.__modelFilename.split("/")[-1]
                    self.__ui.currentModel.setText(filename)
                    self.__data_to_process = filename.split("_")[0]
                    self.__model.setDataToProcess(self.__data_to_process)
                except IndexError:
                    self.__ui.currentModel.setText(self.__modelFilename)
            else:
                print("ERROR: wrong file extension.")
    
    @pyqtSlot()
    def createModel(self):
        """
        Creates a NN model.
        """
        # To avoid ValueError error we have to do a roolback of the model
        self.__model.rollback()
        # Create all layers
        self.__model.createLayers()

        # Compile, fit and evalute the model in a new thread
        self.__thread = ThreadCreateModel(self)
        self.__thread.setTerminationEnabled(True)
        # Start the thread
        self.__thread.start()

        #self.__ui.stopModelCreationBtn.setEnabled(True)
        #if not self.__isStopModelCreationBtnConnected:
        #    self.__ui.stopModelCreationBtn.clicked.connect(self.__thread.stop)
        #    self.__isStopModelCreationBtnConnected = True

    @pyqtSlot(bool)
    def done(self, ret):
        """
        Inform the user of the end of the model training.
        """
        self.__ui.progressBar.setValue(0)

        #self.__ui.stopModelCreationBtn.setEnabled(False)

        if ret:
            filename = self.__data_to_process + datetime.now().strftime("_%H-%M-%S_%d-%m-%y")
            # Save the model with the given name
            self.__model.saveModel(filename)
            print("Model saved.")

            # Allow the use of the model
            self.__isModelLoaded = True

            self.__ui.currentModel.setText(filename)

        print("DONE")

    @pyqtSlot()
    def predictValue(self):
        """
        Predict segmentation or classification of a specific image. Before calling this
        method you have to load an image with the method MainWindow.openFile().
        """
        if self.__isModelLoaded:
            if self.__imgToPredict is not None:
                # Predict the value
                pred = self.__model.predictValue()
                # Show the value
                if self.__data_to_process in ["digits", "animals"]:
                    # Show the predicted value in a QLabel
                    self.__ui.predictedValue.setText(pred)
            else:
                print("ERROR: please, open an image to predict.")
        else:
            print("ERROR: please, create or open a model.")

    @pyqtSlot()
    def plot(self):
        """
        Plots model accuracy and loss.
        """
        # Random data
        if self.__model is not None and self.__model.getHistory() is not None:
            history = self.__model.getHistory()
            print(history.history.keys())
            keys = list(history.history.keys())
            # Training accuracy and loss
            x_acc      = history.history[keys[3]]
            x_loss     = history.history[keys[2]]
            # Validation accuracy and loss
            x_val_acc  = history.history[keys[1]]
            x_val_loss = history.history[keys[0]]

            # Instead of ax.hold(False)
            self.__figure.clear()

            # Create 2 axes
            ax1, ax2 = self.__figure.subplots(1, 2)

            # Plot datas
            ax1.plot(x_acc)
            ax1.plot(x_val_acc)
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("accuracy")
            ax1.set_title("Training and validation accuracy")
            ax2.plot(x_loss)
            ax2.plot(x_val_loss)
            ax2.set_xlabel("epoch")
            ax2.set_ylabel("loss")
            ax2.set_title("Training and validation loss")

            # Refresh canvas
            self.__canvas.draw()
        else:
            print("WARNING: nothing to plot (model and/or model.history is/are None).")

    @pyqtSlot(str)
    def setModel(self, radioBtn):
        """
        Thanks to the selected radio button, instantiate a new neural network model.
        """
        self.__data_to_process = radioBtn
        if radioBtn == "roads":
            self.__model = RoadsModel()
        elif radioBtn == "digits":
            self.__model = DigitsModel()
        elif radioBtn == "animals":
            self.__model = AnimalsModel()
        elif radioBtn == "aerial_buildings":
            self.__model = AerialBuildingsModel()
        elif radioBtn == "aerial_roads":
            self.__model = AerialRoadsModel()
        elif radioBtn == "kidneys":
            self.__model = KidneysModel()

if __name__ == "__main__":
    print("ERROR: this is not the main file of this program.")