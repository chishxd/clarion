# Clarion 🚀

A lightweight FastAPI service for serving machine learning classification models — containerized and ready for local development with S3-compatible storage (MinIO).

---

## 📌 Features

- ✅ FastAPI web server with `/predict` endpoint
- ✅ Loads a serialized ML model (`.pkl`) for live predictions
- ✅ Integrates with S3-compatible storage (MinIO)
- ✅ Local Docker Compose setup for full-stack dev
- ✅ Clean Python package structure (`src/clarion/`)

---

## 🚀 Getting Started

### 1️⃣ Clone the repo

```bash
git clone https://github.com/chishxd/clarion.git
cd clarion
````

---

### 2️⃣ Local development

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .

# Run the API server with hot reload
uvicorn clarion.app:app --reload
```

---

## 🐳 Docker Compose (Recommended)

Run the API server + MinIO together:

```bash
# Make sure you have a .env file with your MinIO credentials:
echo "AWS_ACCESS_KEY_ID=admin" > .env
echo "AWS_SECRET_ACCESS_KEY=root@123" >> .env

# Then build and run containers
docker compose up --build -d
```

* Clarion API: [http://localhost:80](http://localhost:80)
* MinIO Console: [http://localhost:9001](http://localhost:9001)

---

## 🔬 Example API Request

```bash
curl -X POST "http://localhost:80/predict" \
  -H "Content-Type: application/json" \
  -d '{"PassengerId": 123, "Pclass": 1, ...}'
```

**Example Response**

```json
{
  "Survived": 1
}
```

---

## ✅ Tests

Run unit tests with `pytest`:

```bash
pytest
```

---

## ⚙️ Roadmap

* [x] Local FastAPI prediction endpoint
* [x] Model versioning & download from MinIO
* [x] Docker Compose setup
* [ ] Production cloud deployment (On hold — budget-dependent)

---

## 📜 License

Distributed under the MIT License.

---

## ✨ Credits

Clarion is built as part of the *Cloud Computing for Data Science* curriculum — and follows up on [AutoCleanSE](https://github.com/chishxd/autocleanse).

---
## ⚠️ Notes

* Keep `.env` in `.gitignore` — don’t commit credentials.
* For production, use real secrets + a secure MinIO setup.
* Current version is for local dev & student projects — not hardened for prod yet.

---
