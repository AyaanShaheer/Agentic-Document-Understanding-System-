# 🌟 **Agentic Document Understanding System**

### *End-to-End AI Pipeline for Intelligent, Agent-Orchestrated Document Understanding*

🔗 **GitHub Repository:** [https://github.com/AyaanShaheer/Agentic-Document-Understanding-System](https://github.com/AyaanShaheer/Agentic-Document-Understanding-System)

---

<p align="center">
  <img src="https://img.shields.io/github/stars/AyaanShaheer/Agentic-Document-Understanding-System?style=for-the-badge"/>
  <img src="https://img.shields.io/github/forks/AyaanShaheer/Agentic-Document-Understanding-System?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-0db7ed?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

---

# 📸 **Architecture Overview (Diagram)**

> *(Replace the placeholder below with your PNG diagram when ready)*

<p align="center">
  <img src="docs/architecture-diagram.png" alt="Architecture Diagram" width="800"/>
</p>

---

# 🔄 **Pipeline Flowchart (High-Level Flow)**

> *(Replace placeholder with your actual exported flowchart image)*

<p align="center">
  <img src="docs/pipeline-flowchart.png" alt="Pipeline Flowchart" width="800"/>
</p>

---

# 🚀 **Overview**

The **Agentic Document Understanding System** is a production-ready framework that combines:

* Deep Learning
* Vision-Language Models
* Agentic AI (LangGraph)
* FastAPI microservices
* Dockerized deployment

It transforms raw documents into **structured, queryable, intelligent representations** using detection → OCR → layout → multimodal reasoning → LLM querying.

---

# ✨ **Key Features**

### 🔍 Text Detection

Faster-RCNN (ResNet50-FPN) identifies text regions precisely.

### 📝 OCR

Microsoft TrOCR converts detected text regions into high-quality text.

### 📐 Layout Analysis

LayoutLMv3 identifies titles, paragraphs, tables, forms, and more.

### 🤖 Agentic Pipeline

A LangGraph multi-agent system manages:

* OCR agent
* Detection agent
* Layout agent
* Reasoning agent
* Query agent
* Output aggregation

### 🧠 VLM + LLM

* **Gemini Vision** for deep multimodal understanding
* **Groq Llama3.3** for ultra-fast reasoning

### 🧩 REST API

Powered by FastAPI with Swagger UI.
👉 `http://localhost:8000/docs`

### 🐳 Dockerized

Production-ready Dockerfile + docker-compose setup included.

---

# 📁 **Project Structure**

```
Agentic-Document-Understanding-System/
│
├── src/
│   ├── agents/
│   ├── api/
│   ├── detection/
│   ├── recognition/
│   ├── layout/
│   ├── vlm/
│   └── utils/
│
├── outputs/
├── examples/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# ⚙️ **Installation (Local)**

```bash
git clone https://github.com/AyaanShaheer/Agentic-Document-Understanding-System
cd Agentic-Document-Understanding-System

pip install -r requirements.txt

cp .env.example .env  # then add API keys

python run_api.py
```

Swagger UI:
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

# 🐳 **Docker Deployment**

```bash
docker-compose build
docker-compose up -d
```

Logs:

```bash
docker-compose logs -f api
```

---

# 📡 **API Endpoints**

## **1. Health Check**

```
GET /health
```

## **2. Text Detection**

```
POST /detect
file=<image>
```

## **3. Full Analysis**

```
POST /analyze
file=<image>
```

## **4. Query a Document**

```
POST /query
file=<image>
query="Your question"
```

---

# 🧪 **Example Output Section**

## **1. Detection Output (Sample)**

```json
{
  "detected_regions": [
    {
      "bbox": [120, 45, 690, 98],
      "confidence": 0.94
    },
    {
      "bbox": [110, 140, 720, 180],
      "confidence": 0.89
    }
  ]
}
```

---

## **2. OCR Output (Sample)**

```json
{
  "ocr_text": [
    "Invoice Number: INV-2024-0012",
    "Total Amount: ₹58,900"
  ]
}
```

---

## **3. Layout Analysis Output (Sample)**

```json
{
  "layout_elements": [
    {
      "type": "Title",
      "text": "INVOICE"
    },
    {
      "type": "Table",
      "structure": {
        "rows": 5,
        "columns": 4
      }
    }
  ]
}
```

---

## **4. /query Output (Sample)**

**User Question:**
*“What is the total amount on the invoice?”*

```json
{
  "answer": "The total amount on the invoice is ₹58,900."
}
```

---

# 🧾 **API Schema Tables**

## **📌 `/analyze` Request Schema**

| Field  | Type              | Required | Description                   |
| ------ | ----------------- | -------- | ----------------------------- |
| `file` | image (multipart) | Yes      | The document/image to analyze |

---

## **📌 `/analyze` Response Schema**

| Field               | Type   | Description            |
| ------------------- | ------ | ---------------------- |
| `detections`        | array  | Detected text regions  |
| `ocr`               | array  | OCR text results       |
| `layout`            | array  | Layout classifications |
| `summary`           | string | High-level summary     |
| `structured_output` | object | Final JSON             |

---

## **📌 `/query` Request Schema**

| Field   | Type   | Required | Description                      |
| ------- | ------ | -------- | -------------------------------- |
| `file`  | image  | Yes      | Document image                   |
| `query` | string | Yes      | User question about the document |

---

## **📌 `/query` Response Schema**

| Field          | Type   | Description               |
| -------------- | ------ | ------------------------- |
| `answer`       | string | Final LLM answer          |
| `context_used` | array  | OCR + layout context used |

---

# ⚡ **Performance Benchmarks**

| Module        | Avg Time (CPU)    |
| ------------- | ----------------- |
| Detection     | ~2 sec            |
| OCR           | ~1 sec per region |
| Layout        | ~3 sec            |
| Full Pipeline | **10–20 sec**     |

GPU → **8×–12× faster**

---

# 🔧 **Environment Variables**

```
GOOGLE_API_KEY=
GROQ_API_KEY=
DEVICE=cpu
API_HOST=0.0.0.0
API_PORT=8000
```

---

# 🛠️ **Tech Stack**

* **FastAPI**
* **LangGraph Agents**
* **HuggingFace Transformers**
* **Faster-RCNN**, **TrOCR**, **LayoutLMv3**
* **Gemini Vision**
* **Groq Llama3.3**
* **Docker**
* **Python 3.12**

---

# 📝 **License**

MIT License.

---

# ❤️ **Author**

**Ayaan Shaheer**
AI/ML • MLOps • Agents • Document AI
📧 *[gfever252@gmail.com](mailto:gfever252@gmail.com)*



