STEPS TO RUN THIS MODEL IN YOUR LOCAL COMPUTER

MAKE A SPECIFIC FOLDER FOR THIS

DOWNLOAD ALL GIVEN FILES IN THAT FOLDER

USE POWERSHELL TO RUN ALL COMMANDS

IN STEP 0 USE POWERSHELL AS ADMINISTRATOR

Step 0 — Install Python 3.11   
``` cmd =  winget install Python.Python.3.11 ```

downloads and installs Python 3.11.9 automatically 

Step 1 — Navigate to project folder
```For Example =  "C:\Users\vinit\OneDrive\Desktop\AI Driven Waste Classifier"```

moves PowerShell into your project folder

Step 2 — Create venv
```cmd = py -3.11 -m venv venv```

creates a clean isolated Python 3.11 environment inside your project folder

Step 3 — Activate venv
```cmd = venv\Scripts\activate.ps1```

switches PowerShell to use your project's Python instead of system Python — you'll see (venv) appear

Step 4 — Install packages
```cmd = pip install -r requirements.txt```

installs all packages at exact same versions from requirements.txt — no need to type them manually

Step 5 — Verify
```cmd = python -c "import tensorflow as tf; import cv2; print('TF:', tf.__version__); print('OpenCV:', cv2.__version__)"```


confirms everything installed correctly before running

**Expected correct output:**
```
TF: 2.16.1
OpenCV: 4.11.0
```

if you see these exact versions — everything is good, proceed to Step 6
if you see any error — packages didn't install correctly, re-run Step 4

Step 6 — Run camera
```cmd = python camera.py```


loads your trained model and opens live webcam — press `Q` to quit
```
## When to run each step:

| Step | When to run |
|---|---|
| Step 0 | Only once |
| Steps 1–5 | Only once per new project setup |
| Step 6 | Every time you want to run the AI |


## Your project folder should look like:

AI Driven Waste Classifier/
├── waste_model.h5       ← trained AI model
├── camera.py            ← live camera script
├── requirements.txt     ← all package versions
├── .python-version      ← python version reference
└── venv/                ← isolated environment
