# 🌱 Bean Leaf Doctor: an end-to-end ML pipeline

A farmer takes a photo of a bean leaf, and the app says whether it is **healthy**, has **bean rust**, or has **angular leaf spot**, with advice on what to do next. Beans are a staple crop for Rwandan smallholder farmers. Catching these diseases early protects household income. This project continues my *Ascend Rwanda* mission from the Intro to ML summative (economic opportunity).

| | Link |
|---|---|
| 🎥 Video demo | **TODO: YouTube link** |
| 🌍 Live app | **TODO: Hugging Face Space URL** |
| 📓 Notebook | [`notebook/bean_leaf_disease.ipynb`](notebook/bean_leaf_disease.ipynb) |

## What the pipeline does
1. **Data acquisition**: downloads the iBean dataset (Makerere AI Lab, 3 classes, about 1,300 phone photos from East African farms).
2. **Data processing**: validate, convert to RGB, resize to 224×224, scale to [-1, 1] inside the model, augment during training (`src/preprocessing.py`).
3. **Model creation**: MobileNetV2 pre-trained on ImageNet, plus a new classification head with dropout and L2, Adam, early stopping and fine-tuning (`src/model.py`).
4. **Model testing**: accuracy, loss, precision, recall, F1, ROC-AUC and a confusion matrix (notebook and `/metrics`).
5. **Retraining**: uploaded images are preprocessed and **saved to a SQLite database**. The *Retrain* button loads **our own saved model** as the starting point and fine-tunes it on the new images plus a sample of old ones. It keeps the new model only if test accuracy does not drop (`src/model.py::retrain`).
6. **API**: FastAPI (`api/main.py`), with interactive docs at `/docs`.
7. **UI**: dashboard at `/` showing model up-time, 4 data visualisations, a predict box, bulk upload and a retrain button.

## Results (test set, 128 unseen images)
| Model | Accuracy | Loss | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline CNN (no optimisation) | 0.750 | 0.620 | 0.790 | 0.751 | 0.733 | 0.909 |
| MobileNetV2 transfer learning (optimised) | **0.891** | **0.249** | **0.896** | **0.891** | **0.889** | **0.985** |

## Project structure
```
bean-mlop/
├── README.md
├── notebook/bean_leaf_disease.ipynb   # preprocessing, training, evaluation, retraining demo
├── src/
│   ├── preprocessing.py               # image cleaning + colour features
│   ├── model.py                       # build, train, evaluate, retrain
│   ├── prediction.py                  # load model + predict one image
│   └── database.py                    # SQLite: uploads + retraining history
├── api/main.py, api/static/index.html # FastAPI + web dashboard
├── data/train/, data/test/            # images by class folder
├── models/bean_model.keras (+ .h5, metrics.json)
├── locust/locustfile.py               # load test
├── Dockerfile, docker-compose.yml, nginx.conf
└── requirements.txt
```

## How to set it up

### 1. Train the model (Google Colab, free GPU)
1. Open `notebook/bean_leaf_disease.ipynb` in [Google Colab](https://colab.research.google.com) (*File → Upload notebook*).
2. *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all* (about 15 minutes).
3. The last cell downloads `bean-mlop-outputs.zip`. Copy its `data/train`, `data/test` and `models` folders into this repository.
4. If Colab's TensorFlow version (printed in the first cell) is not 2.21, change `tensorflow-cpu==` in `requirements.txt` to match.

### 2. Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --port 7860
```
Open http://localhost:7860 (dashboard) or http://localhost:7860/docs (API docs).

### 3. Run with Docker
```bash
docker build -t bean-mlop .
docker run -p 7860:7860 bean-mlop
```

### 4. Deploy to the cloud (Hugging Face Spaces, free)
1. Create a Space at https://huggingface.co/new-space, choose **Docker → Blank**, hardware **CPU basic (free)**.
2. Push this repository to the Space (it builds from the `Dockerfile` and serves on port 7860).
3. Add this block to the very top of the README in the Space:
```
---
title: Bean Leaf Doctor
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---
```

## Flood request simulation (Locust)
Several copies of the API run behind an nginx load balancer, and Locust sends many prediction requests at once.
```bash
pip install -r locust/requirements.txt
docker compose up --build -d --scale api=1        # then repeat with api=2 and api=3
locust -f locust/locustfile.py --host http://localhost:8080 --users 50 --spawn-rate 10 \
       --run-time 1m --headless --csv locust/results/containers_1 --html locust/results/containers_1.html
docker compose down
```

| Containers | Users | Requests/s | Median (ms) | 95th percentile (ms) | Failures |
|---|---|---|---|---|---|
| 1 | 50 | TODO | | | |
| 2 | 50 | TODO | | | |
| 3 | 50 | TODO | | | |

**What it shows:** TODO, for example "with one container, response time grows as requests queue up; adding containers spreads the load, so latency drops and requests per second rise."

## Dataset credit
iBean dataset, Makerere AI Lab, Uganda (MIT License): https://github.com/AI-Lab-Makerere/ibean
