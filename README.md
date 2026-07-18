# GeoQuery

GeoQuery is a web application that enables natural language querying of RGB aerial images using image captioning, object detection, and visual question answering.

## Clone the Repository

```bash
git clone <repository-url>
cd GeoQuery
```

Open the project in your preferred IDE.

---

# Create a Virtual Environment

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

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Verify Installation

```bash
python --version
pip --version
```

---

# Running the Application

The backend and frontend must be run simultaneously in separate terminals.

## Terminal 1 – Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

---

## Terminal 2 – Frontend

```bash
cd frontend
python3 -m http.server 5500
```

---

Open your browser and navigate to:

```
http://localhost:5500
```

---

# Using GeoQuery

1. Drag and drop an RGB image (or click to upload).
2. Wait for the image to finish processing.
3. The application will display:
   - Annotated image
   - Image caption
   - Detected objects
4. Ask questions such as:
   - Describe the image.
   - What objects are present?
   - How many buildings are there?
   - Is there a road?
   - Highlight all vehicles.

---

# Supported Object Classes

GeoQuery currently supports detection of:

- Building
- Road
- Vehicle
- Vegetation
- Water Body
- Open Ground
