import sys
import logging
from PySide6.QtWidgets import QApplication
from dotenv import load_dotenv

from .gui.main_window import MainWindow

def main() -> None:
    """Application entry point."""
    # Load environment variables from .env file
    load_dotenv()

    # Configure logging centrally (DEBUG level for diagnostics)
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, force=True) # Force config
    logging.debug("Central logging configured in main.py") # Test message

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
