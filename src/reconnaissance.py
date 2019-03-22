try:
    import sys
    from ui import MainWindow
    # Qt 5 imports
    from PyQt5.QtWidgets import QApplication
except ImportError as err:
    exit(err)

def main():
    app = QApplication(sys.argv)
    myWindow = MainWindow(950, 750)
    exit(app.exec_())

if __name__ == "__main__":
    main()