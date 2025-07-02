# Clarion 🚀

A lightweight FastAPI service for serving machine learning classification models — containerized and ready for local or cloud deployment.

---

## 📌 Features

- ✅ **FastAPI** web server with `/predict` endpoint
- ✅ Loads a serialized ML model (`.pkl`) for live predictions
- ✅ Integrates with S3-compatible storage (MinIO) for model assets
- ✅ Containerized with **Docker** for portability
- ✅ Clean Python package structure (`src/`, `models/`, `tests/`)

---

## 🚀 Getting Started

### 1️⃣ Clone the repo

```bash
git clone https://github.com/chishxd/clarion.git
cd clarion
````

### 2️⃣ Local development

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package in editable mode
pip install -e .

# Run the API server with hot reload
uvicorn clarion.app:app --reload
```

---

## 🐳 Docker

Build and run the containerized API:

```bash
# Build image
docker build -t clarion-api .

# Run container
docker run -p 8000:8000 clarion-api
```

Access the API at [http://localhost:8000](http://localhost:8000).

---

## 🔬 Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.2, "feature2": 0.8, "...": "..."}'
```

**Sample Response**

```json
{
  "prediction": 1
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
* [x] Model file versioning (`models/`)
* [x] S3-compatible storage tested with MinIO
* [x] Containerized with Docker
* [ ] Cloud deployment (Planned)

---

## 📜 License

Distributed under the **MIT License**.

---

## ✨ Credits

Clarion is built as part of the **Cloud Computing for Data Science** curriculum.
It follows up on the [AutoCleanSE](https://github.com/chishxd/autocleanse) project for robust data preprocessing.
