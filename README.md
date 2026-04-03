# Data Science Career Assistant Chatbot

## Project Overview

The Data Science Career Assistant Chatbot is a beginner-friendly full-stack AI project that helps users explore careers in data science. It answers career-related questions, recommends important skills, shares interview preparation tips, provides a learning roadmap, and responds gracefully when a query falls outside its knowledge base.

The project uses a FastAPI backend and a simple frontend built with HTML, CSS, and JavaScript. NLP response matching is powered by TF-IDF and cosine similarity using scikit-learn.

## Features

- FastAPI backend with a `POST /chat` API endpoint
- NLP chatbot logic using TF-IDF and cosine similarity
- Clean modular backend structure
- Frontend chat interface with modern styling
- Skill recommendations for Python, SQL, machine learning, statistics, and tools
- Interview preparation guidance and project suggestions
- Learning roadmap for aspiring data science professionals
- Graceful fallback response for unknown questions
- Deployment-ready for Render

## Tech Stack

- Python
- FastAPI
- Uvicorn
- scikit-learn
- NumPy
- pandas
- HTML
- CSS
- JavaScript

## Project Structure

```text
chatbot_project/
│
├── backend/
│   ├── main.py
│   ├── chatbot.py
│   ├── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│
├── README.md
└── .gitignore
```

## How To Run Locally

### 1. Move into the project folder

```bash
cd chatbot_project
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install requirements

```bash
pip install -r backend/requirements.txt
```

### 4. Run the FastAPI server using Uvicorn

```bash
uvicorn backend.main:app --reload
```

### 5. Open the application

Visit:

```text
http://127.0.0.1:8000
```

## API Example

### Endpoint

```http
POST /chat
```

### Sample Request Body

```json
{
  "message": "What skills do I need for data science?"
}
```

### Sample Response

```json
{
  "response": "Core skills for data science are Python, SQL, statistics, machine learning..."
}
```

## GitHub Setup

Use these commands after creating a new empty GitHub repository, for example `data-science-career-assistant-chatbot`.

### Initialize git repo

```bash
cd chatbot_project
git init
git add .
git commit -m "Initial commit: Data Science Career Assistant Chatbot"
```

### Connect to GitHub

Replace `YOUR_USERNAME` with your GitHub username:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/data-science-career-assistant-chatbot.git
git push -u origin main
```

## Render Deployment Guide

### 1. Create a new Web Service

- Sign in to [Render](https://render.com/)
- Click `New +`
- Select `Web Service`
- Connect your GitHub account
- Choose your repository: `data-science-career-assistant-chatbot`

### 2. Configure the service

- Environment: `Python 3`
- Root Directory: leave blank
- Build Command:

```bash
pip install -r backend/requirements.txt
```

- Start Command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 10000
```

### 3. Deploy

- Click `Create Web Service`
- Wait for the build and deploy logs to finish
- Open the generated Render URL to use the chatbot

### Render Troubleshooting

- If Render says it cannot find `chatbot_project/backend/requirements.txt`, your service is using the wrong path.
- This repository already has `backend/` at the repo root, so the correct build file is `backend/requirements.txt`.
- The correct start target is `backend.main:app`.
- You can also deploy using the included `render.yaml` file so Render auto-fills the correct commands.

## Screenshots

Add screenshots here after running the app locally or deploying it.

- `Homepage screenshot goes here`
- `Chat conversation screenshot goes here`

## Author

**Author:** Varun  
**Project:** Data Science Career Assistant Chatbot

## Notes

- The chatbot uses NLP-based similarity matching rather than hardcoded if-else replies.
- You can expand the knowledge base in `backend/chatbot.py` to support more career topics.
- Because the frontend is served by FastAPI, the app is easy to deploy as a single Render web service.
