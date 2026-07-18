# GeoQuery
To run the model, clone the repository to your local machine.
Open the folder in your preferred IDE.

## Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows (Command Prompt)

```cmd
python -m venv venv
venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Verify Installation

```bash
python --version
pip --version
```

# To run the full stack:
## A) Run the backend:
in one IDE terminal, run:

```
cd backend
uvicorn main:app --reload --port 8000
```

## B) Run the frontend:
Immediately after, in another terminal, run:

```
cd ..
cd frontend
python3 -m http.server 5500
```

Then go to your browser and open : http://localhost:5500

## Instructions for the frontend on the web browser:

1. Drag & drop an image.
2. Wait for upload.
3. You should see:
  a) Annotated image
  b) Caption
  c) Detected objects
4. Ask questions like:
  -> Describe the image.
  -> What objects are present?
  -> How many people are there?
