# CareerSphere AI 2.0 

A highly intelligent, data-driven web platform that supports students from all domains (Science, IT/Engineering, Commerce, Arts/Humanities) by providing personalized predictions for:
*   **Career Readiness**
*   **Mental Health Risk** 
*   **Burnout Likelihood**

## System Architecture

*   **Frontend**: React (Vite-based) + Recharts + Lucide-React
*   **Backend**: Python FastAPI
*   **Machine Learning**: Scikit-Learn + XGBoost Classifier (3 distinct models)
*   **Database**: SQLite/PostgreSQL (SQLAlchemy managed)

---

## Running the Application Locally

### 1. Start the Backend

Require Python 3.9+.

```bash
cd backend
python -m venv venv
# Windows: venv\\Scripts\\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

# Run FastAPI Server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`
Swagger UI Documentation: `http://localhost:8000/docs`

### 2. Start the Frontend

Require Node.js v16+.

```bash
cd frontend
npm install
npm run dev
```

The React Vite server will typically start at `http://localhost:5173`. Open this in the browser to view the dynamic assessment platform.

---

## Deployment Guide (Docker & Cloud)

### Containerization (Production)

Create a `docker-compose.yml` to orchestrate both the frontend and backend. 

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/careersphere

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend

  db:
    image: postgres:13
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: careersphere
```

### Cloud Providers
1. **Frontend (Vercel / Netlify)**: Connect your GitHub Repo, configure the build command as `npm run build` and output directory as `dist`. Set the generic `API_BASE` environment variable to your deployed backend URL.
2. **Backend (Render / App Platform / AWS EC2)**: Deploy the FastAPI app. Ensure your serialized models (`.pkl` files inside `backend/models`) are included in the build or downloaded from an S3 bucket during continuous integration.
3. **Database (Supabase / RDS)**: Provision a production database and update the `SQLALCHEMY_DATABASE_URL` in `backend/app/db/database.py` via your remote `.env` variables.

## Intelligent Personalization
To customize further, the `Chatbot` heuristic engine is located in `backend/app/api/chat.py`, and domain-based logic is located at the frontend in `src/App.jsx`.
