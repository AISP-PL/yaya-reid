# Generate all UI files
export PYUIC_ARGS="--import-from=."
pyuic5 ${PYUIC_ARGS} MainWindow.ui > Ui_MainWindow.py
