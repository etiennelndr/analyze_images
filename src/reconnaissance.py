try:
    import sys
    # Insert path to mainwindow file
    sys.path.insert(0, "ui")
    
    # Qt 5 imports
    from PyQt5.QtWidgets import QApplication
    from mainwindow import MainWindow
except ImportError as err:
    exit(err)

def main():
    app = QApplication(sys.argv)
    myWindow = MainWindow(950, 750)
    exit(app.exec_())

if __name__ == "__main__":
    main()