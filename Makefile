# --- Configuration ---
VENV := .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/bin/pip

# --- Default Target ---
.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  setup   : Create virtual environment and install dependencies"
	@echo "  train   : Run the training script (train.py)"
	@echo "  predict : Run the prediction script (predict.py)"
	@echo "  clean   : Remove virtual environment and cached files"
	@echo "  help    : Show this help message (default)"

# --- Environment Setup ---
setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	@echo "Checking for virtual environment..."
	@test -d $(VENV) || (echo "Creating venv..." && python3 -m venv $(VENV))
	@echo "Updating pip and installing requirements..."
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt
	@touch $(VENV)/bin/activate
	@echo "Setup complete. Use 'source $(VENV)/bin/activate' to enter the environment manually."

# --- Project Execution ---
train: setup
	$(PYTHON) train.py

predict: setup
	$(PYTHON) predict.py

# --- Cleanup ---
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Environment removed."

.PHONY: setup train predict clean help