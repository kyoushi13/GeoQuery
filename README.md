# GeoQuery

GeoQuery is a web application that lets users interact with RGB aerial images using natural language. Instead of manually inspecting an image, users can simply upload it and ask questions such as *"How many buildings are there?"*, *"Is there a road?"*, or *"Highlight all the vehicles."*

The project combines image captioning, object detection, and visual question answering (VQA) to create an interactive way of exploring aerial imagery.

---

## Features

- Upload RGB aerial images through a simple drag-and-drop interface
- Automatically generate a caption describing the scene
- Detect and annotate key objects in the image
- Ask questions about the uploaded image in natural language
- Support for multi-turn conversations with context retention
- Click on any detected object to view additional information
- Highlight specific object classes directly on the image
- Export the complete analysis and conversation as a PDF report

---

## Clone the Repository

```bash
git clone <repository-url>
cd GeoQuery
```
Open the project in your preferred IDE.

### Or download the zip file

Unzip the file and open the folder in your preferred IDE.
Then navigate to the folder via the terminal.

---

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
.\venv\Scripts\Activate.ps1
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Verify Installation

```bash
python --version
pip --version
```

---

# Running the Application

The backend and frontend should be started in separate terminals.

### Terminal 1 – Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Ensure you are able to see:
```
INFO:     Started server process [23488]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Terminal 2 – Frontend

```bash
cd frontend
python3 -m http.server 5500
```

Once both are running, open:

```
http://localhost:5500
```

---

# Using GeoQuery

### 1. Upload an Image

Upload an RGB aerial image by dragging and dropping it into the upload area or by selecting it manually.

Once uploaded, GeoQuery automatically:

- generates an image caption,
- detects supported objects,
- creates an annotated image, and
- prepares the image for natural language querying.

---

### 2. Explore the Scene

After processing is complete, you'll see:

- the annotated image,
- the generated caption,
- and a list of detected objects.

You can then ask questions such as:

- Describe the image.
- What objects are present?
- How many buildings are there?
- Is there a road?
- Highlight all vehicles.
- Count the trees.
- Are there any water bodies?

---

### 3. Inspect Individual Objects

Every detected object is represented by a clickable bounding box.

Clicking on a box displays additional information about that object, including:

- object class,
- confidence score,
- estimated size, and
- dominant colour.

---

### 4. Continue the Conversation

GeoQuery remembers the context of your conversation, so follow-up questions work naturally.

For example:

```
Highlight all buildings.

How many are there?

Which one has the highest confidence?

Highlight the vehicles now.
```

There's no need to repeat the full query every time—the application keeps track of the ongoing conversation during the session.

---

### 5. Export Your Results

At any point, click **Download PDF Report** to generate a report containing:

- the annotated image,
- generated caption,
- detected objects,
- and the complete conversation history.

This makes it easy to save or share your analysis.

---

# Supported Object Classes

GeoQuery currently detects:

- Building
- Road
- Vehicle
- Vegetation
- Water Body
- Open Ground

---

# Project Structure

```
GeoQuery
│
├── backend
│   ├── main.py
│   ├── detect.py
│   ├── caption_vqa.py
│
├── frontend
│   └── index.html
│
├── requirements.txt
└── README.md
```

---

# Tech Stack

- FastAPI
- PyTorch
- Hugging Face Transformers (BLIP)
- Ultralytics YOLO-World
- Pillow
- ReportLab
- HTML, CSS and JavaScript

---

# Current Limitations

- Designed specifically for RGB aerial imagery.
- Supports a fixed set of object classes.
- Caption generation speed depends on the available hardware.
- Detection quality is limited by the underlying YOLO-World model.

---

## Future Improvements

Some ideas we'd like to explore in the future include:

- support for additional object classes,
- better aerial-image-specific object detection,
- segmentation-based highlighting,
- geospatial coordinate support,
- and batch image processing.

---
