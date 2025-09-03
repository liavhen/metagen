import os

# -----------------------------------------------------
# General
# -----------------------------------------------------
PROJECT_DIR = '.'
STORAGE_DIR = './storage'
os.makedirs(STORAGE_DIR, exist_ok=True)

# -----------------------------------------------------
# Results
# -----------------------------------------------------
SAMPLINGS_DIR = os.path.join(STORAGE_DIR, 'samplings')
RUNS_DIR = os.path.join(STORAGE_DIR, 'runs')
OPTIM_DIR = os.path.join(STORAGE_DIR, 'optimization') #
os.makedirs(OPTIM_DIR, exist_ok=True)

# -----------------------------------------------------
# Assets
# -----------------------------------------------------
ASSETS_DIR = os.path.join(STORAGE_DIR, 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)
# -----------------------------------------------------


# -----------------------------------------------------
# Data
# -----------------------------------------------------
DATA_DIR = os.path.join(STORAGE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
# -----------------------------------------------------

