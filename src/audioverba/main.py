import sys
from PySide6.QtWidgets import QApplication
from dotenv import load_dotenv

from .gui.main_window import MainWindow

def main() -> None:
    """Application entry point."""
    # Load environment variables from .env file
    load_dotenv()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
