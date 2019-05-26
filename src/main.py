try:
    import sys
    from ui import MainWindow
    # Qt 5 imports
    from PyQt5.QtWidgets import QApplication
except ImportError as err:
    exit(err)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Analyze images")
    my_window = MainWindow(950, 750)
    app.setActiveWindow(my_window)
    exit(app.exec_())


if __name__ == "__main__":
    main()