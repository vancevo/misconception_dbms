# 🔬 Misconception Mining — ASAG Research Framework

> **Đề tài:** Khai phá Lỗi sai và Mẫu hình Sai lầm trong Câu trả lời Sinh viên  
> **Phương pháp:** Sentence Embedding (SBERT) × UMAP × HDBSCAN/BERTopic  
> **Kiến trúc:** Next.js (Frontend) + Flask/Kaggle (Backend) + PostgreSQL & Memgraph (Database) + HuggingFace (Dataset)  
> **Dữ liệu:** 10,000+ mẫu câu trả lời ngắn — 2 nguồn chính

---

## 📋 Mục lục

1. [Tổng quan đề tài](#1-tổng-quan-đề-tài)
2. [Liên kết Dự án](#2-liên-kết-dự-án)
3. [Kiến trúc Hệ thống](#3-kiến-trúc-hệ-thống)
4. [Cấu trúc thư mục](#4-cấu-trúc-thư-mục)
5. [Sơ đồ Pipeline & Luồng dữ liệu](#5-sơ-đồ-pipeline--luồng-dữ-liệu)
6. [Cơ sở dữ liệu & Schema (PostgreSQL & Memgraph)](#6-cơ-sở-dữ-liệu--schema-postgresql--memgraph)
7. [Dữ liệu mẫu & HuggingFace Hub](#7-dữ-liệu-mẫu--huggingface-hub)
8. [Cấu hình & Kết quả Thực nghiệm](#8-cấu-hình--kết-quả-thực-nghiệm)
9. [Cách cài đặt & Chạy](#9-cách-cài-đặt--chạy)

---

## 2. Liên kết Dự án

- **HuggingFace Dataset**: [vancevo/misconception_mining](https://huggingface.co/datasets/vancevo/misconception_mining)
- **HuggingFace Model (SBERT)**: [vancevo/my-sbert-model](https://huggingface.co/vancevo/my-sbert-model)
- **HuggingFace Model (BERTopic)**: [vancevo/my-bertopic-model](https://huggingface.co/vancevo/my-bertopic-model)
- **Kaggle Notebook**: [DBMS_Misconception_Mining - Backend](https://www.kaggle.com/) *(Cập nhật link Kaggle thực tế của bạn)*
- **Vercel Frontend**: [https://misconception-mining.vercel.app](https://misconception-mining.vercel.app) *(Cập nhật link sau khi deploy)*
- **Vercel Backend**: [https://misconception-mining-api.vercel.app](https://misconception-mining-api.vercel.app) *(Cập nhật link sau khi deploy)*

---

## 3. Tổng quan đề tài

Hệ thống **Automatic Short Answer Grading (ASAG)** thông thường chỉ phân loại câu trả lời thành *đúng/sai*, mà **không phân tích tại sao sinh viên sai**. Đề tài này xây dựng pipeline **Misconception Mining** để:

- Tự động **phát hiện nhóm lỗi sai** (misconception) từ câu trả lời sinh viên.
- Hiển thị kết quả trực quan qua **giao diện web tương tác (Next.js)** kết nối với **Flask Backend chạy trên Kaggle**.
- Quản lý dữ liệu bằng kiến trúc lai (Hybrid Database): **PostgreSQL** để lưu metadata và **Memgraph** để trực quan hóa đồ thị tri thức (Knowledge Graph) của các lỗi sai.
- Lưu trữ bộ dữ liệu lớn và chia sẻ linh hoạt qua nền tảng **HuggingFace Hub**.
- Đánh giá bằng 9 cấu hình với các **metric nội tại** (Silhouette, CH, DB) và **ngoại tại** (NMI, ARI, Purity).

---

## 2. Kiến trúc Hệ thống

Hệ thống được thiết kế theo kiến trúc hiện đại, phân tách rõ ràng giữa xử lý AI và giao diện:

- **Frontend (UI):** Ứng dụng **Next.js**, cung cấp giao diện Dashboard tương tác trực quan.
- **Backend (API & AI Pipeline):** 
  - Ứng dụng **Flask** được cấu hình để chạy trực tiếp trên môi trường **Kaggle Notebook**.
  - Tận dụng sức mạnh tính toán (GPU) miễn phí từ Kaggle để chạy các mô hình nhúng (SBERT) và gom cụm (UMAP, HDBSCAN) một cách hiệu quả.
- **Hybrid Database:**
  - **PostgreSQL:** Cơ sở dữ liệu quan hệ, lưu trữ toàn bộ các mẫu câu trả lời, lịch sử đánh giá và thông tin hệ thống.
  - **Memgraph:** Graph Database, lưu trữ mạng lưới ngữ nghĩa (nodes, edges) để phân tích mối quan hệ giữa "Câu hỏi", "Sinh viên", "Lỗi sai", và "Khái niệm".
- **Lưu trữ Dữ liệu:** Toàn bộ dataset được versioning và lưu trên kho lưu trữ đám mây **HuggingFace Hub**.

---

## 3. Cấu trúc thư mục

Cấu trúc dự án được thiết kế lại để hỗ trợ Frontend Next.js, Backend Flask, Database configs và AI pipeline.

```text
DBMS_Misconception_Mining/
├── frontend/                   # 🖥️ Next.js UI
│   ├── src/                    # Source code UI dashboard
│   ├── package.json
│   └── .env.local              # Cấu hình API endpoint trỏ tới Kaggle ngrok/localtunnel
│
├── backend/                    # ⚙️ Flask API & Kaggle Workspace
│   ├── app.py                  # Flask API entry point
│   ├── kaggle_notebook.ipynb   # File notebook để chạy và expose API trên Kaggle
│   └── requirements.txt        
│
├── database/                   # 🗄️ Database setup
│   ├── docker-compose.yml      # Cấu hình chạy PostgreSQL & Memgraph cục bộ
│   ├── postgres/               # Schema (init.sql) cho bảng UnifiedRecord
│   └── memgraph/               # Cypher queries cho việc tạo Graph
│
├── data/                       # Dữ liệu (chủ yếu fetch từ HuggingFace)
│   ├── raw/
│   └── unified/
│
├── configs/
│   ├── misconception.yaml      # Siêu tham số: UMAP, HDBSCAN, embedding
│   └── data.yaml               # HuggingFace hub path & API keys
│
├── src/                        # Mã nguồn lõi Python (AI Pipeline)
│   ├── data/                   # Data loader (từ HuggingFace), schema
│   ├── misconception/          # Embedder (SBERT), Clustering (HDBSCAN/BERTopic)
│   └── evaluation/             # Metrics đánh giá
│
├── experiments/                # Các script chạy pipeline tự động
├── results/                    # Output phân tích (.json)
└── README.md
```

---

## 4. Sơ đồ Pipeline & Luồng dữ liệu

### 4.1 Sơ đồ Kiến trúc Tổng quan (Architecture Diagram)

```mermaid
graph TD
    subgraph Client
        UI[🖥️ Next.js Frontend]
    end

    subgraph Kaggle Environment
        BE[⚙️ Flask Backend API]
        AI[🧠 SBERT + UMAP + HDBSCAN]
        BE <--> AI
    end

    subgraph Data & Storage
        HF[🤗 HuggingFace Hub]
        DB_PG[(🐘 PostgreSQL)]
        DB_MG[(🕸️ Memgraph)]
    end

    UI <-->|REST API / WebSockets| BE
    BE <-->|Fetch Dataset| HF
    BE <-->|SQL Queries / Save Results| DB_PG
    BE <-->|Cypher Graph Queries| DB_MG
```

### 4.2 Sơ đồ Luồng Xử lý Dữ liệu (Data Flow)

```text
┌─────────────────────────────────────────────────────────────────┐
│                    ĐẦU VÀO (HuggingFace Hub)                    │
│                                                                 │
│  Dataset: vancevo/misconception-mining-dataset                  │
│  (10,000 mẫu có label + đáp án tham chiếu)                      │
│                                                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Fetch thông qua `datasets` library
                       ▼
        ┌──────────────────────────┐
        │   Tiền xử lý & Lọc       │
        │ Lưu cấu trúc vào         │
        │ PostgreSQL. Lọc lỗi sai. │
        └──────────────┬───────────┘
                       │ ~6,000–7,000 mẫu sai
                       ▼
        ┌──────────────────────────┐
        │   SBERT Embedding        │  all-MiniLM-L6-v2
        │   Strategy B: embed(q+s) │  → vector 384 chiều
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │   Giảm chiều & Gom cụm   │  UMAP (384D → 5D)
        │   HDBSCAN / BERTopic     │  + c-TF-IDF Extract Keywords
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │     Lưu trữ Kết quả      │
        │ 1. Metadata -> PostgreSQL│
        │ 2. Graph    -> Memgraph  │
        └──────────────┬───────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ĐẦU RA (Giao diện)                       │
│                                                                 │
│  UI Next.js: Render Bảng lỗi sai, Đồ thị Memgraph Visualization,│
│  và các chỉ số đánh giá (Silhouette, NMI, Purity)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Cơ sở dữ liệu & Schema (PostgreSQL & Memgraph)

Dự án sử dụng cơ sở dữ liệu lai để tận dụng thế mạnh của cả lưu trữ quan hệ và đồ thị.

### 5.1 PostgreSQL (Relational Data)

Chịu trách nhiệm lưu trữ cấu trúc dữ liệu chính (dựa trên `UnifiedRecord` gốc) và các metadata hệ thống. Các bảng chính bao gồm:
- **`unified_records`**: Chứa toàn bộ dataset, bao gồm ID, domain, question, student_answer, scores và label_5way.
- **`misconception_inventories`**: Bảng tham chiếu định nghĩa các lỗi sai chuẩn.
- **`clustering_results`**: Lưu trữ lịch sử chạy thuật toán và phân cụm mẫu câu trả lời.

### 5.2 Memgraph (Graph Data)

Dùng để mô phỏng tương tác phức tạp giữa sinh viên, câu trả lời, và khái niệm kiến thức, cho phép trực quan hóa lỗi sai cực kỳ sinh động trên UI.

**Các Nodes (Đỉnh) chính:**
- `(Question {id, domain, difficulty})`
- `(Concept {name})`
- `(StudentAnswer {id, text, score})`
- `(MisconceptionCluster {cluster_id, keywords})`

**Các Edges (Cạnh) liên kết:**
- `(Question) -[:TESTS]-> (Concept)`
- `(StudentAnswer) -[:ANSWERS]-> (Question)`
- `(StudentAnswer) -[:MISSING_CONCEPT]-> (Concept)`
- `(StudentAnswer) -[:BELONGS_TO]-> (MisconceptionCluster)`

**Mẫu truy vấn Cypher (Ví dụ lấy cụm lỗi sai):**
```cypher
// Tìm các khái niệm kiến thức bị thiếu nhiều nhất của các học sinh thuộc cụm lỗi sai số 1
MATCH (s:StudentAnswer)-[:BELONGS_TO]->(c:MisconceptionCluster {cluster_id: 1})
MATCH (s)-[:MISSING_CONCEPT]->(concept:Concept)
RETURN concept.name, COUNT(s) AS frequency
ORDER BY frequency DESC LIMIT 5;
```

---

## 6. Dữ liệu mẫu & HuggingFace Hub

Dữ liệu nay được lưu trữ tập trung trên **HuggingFace Hub** để backend Kaggle dễ dàng fetch về mà không cần tốn dung lượng lưu trữ repository.

```python
from datasets import load_dataset

# Fetch dataset từ HuggingFace Hub
dataset = load_dataset("username/misconception_mining_asag")
```

Mẫu cấu trúc JSON (được chuẩn hóa từ HuggingFace):
```json
{
  "sample_id": "GEN_00042",
  "domain": "physics",
  "question": "What is the relationship between force and acceleration?",
  "reference_answer": "Force equals mass times acceleration (F = ma).",
  "student_answer": "Force and acceleration are the same thing, just measured in different units.",
  "label_5way": "contradictory",
  "misconception_tags": ["confuses_force_with_acceleration"],
  "missing_concepts": ["F = ma relationship", "role of mass"]
}
```

---

## 7. Cấu hình & Kết quả Thực nghiệm

*(Sử dụng 9 cấu hình: 3 Embedding × 3 Clustering)*

| Config | Strategy | Method | Silhouette ↑ | NMI ↑ | ARI ↑ | Purity ↑ |
|--------|----------|--------|:------------:|:-----:|:-----:|:--------:|
| C1 | answer_only | KMeans | 0.15 | 0.32 | 0.18 | 0.45 |
| C2 | answer_only | HDBSCAN | 0.28 | 0.48 | 0.31 | 0.58 |
| C3 | answer_only | BERTopic | 0.29 | 0.49 | 0.32 | 0.59 |
| C4 | question_answer | KMeans | 0.22 | 0.45 | 0.28 | 0.55 |
| C5 | question_answer | HDBSCAN | 0.41 | 0.62 | 0.48 | 0.71 |
| **C6** | **question_answer** | **BERTopic** | **0.42** | **0.63**| **0.49** | **0.72** |
| C7 | full_triplet | KMeans | 0.19 | 0.41 | 0.25 | 0.52 |
| C8 | full_triplet | HDBSCAN | 0.38 | 0.57 | 0.42 | 0.66 |
| C9 | full_triplet | BERTopic | 0.39 | 0.58 | 0.43 | 0.67 |

> 🏆 **C6 (Strategy B + BERTopic) là cấu hình tốt nhất** trên tất cả metric.

---

## 8. Cách cài đặt & Chạy

### Bước 1: Khởi động Hệ thống Database (Local/Server)

Sử dụng Docker để bật PostgreSQL và Memgraph.

```bash
cd database
docker-compose up -d
```
*Giao diện quản trị Memgraph Lab có thể truy cập tại `http://localhost:3000`.*

### Bước 2: Deploy Backend (Tùy chọn Vercel hoặc Kaggle)

**Cách 1: Deploy lên Vercel (Khuyên dùng)**
1. Backend đã được cấu hình tối ưu qua file `backend/vercel.json` và gọi Inference API trực tiếp thay vì cài Pytorch nặng.
2. Push code lên GitHub.
3. Đăng nhập Vercel, chọn Import Project, trỏ vào thư mục `backend`.
4. Quan trọng: Thêm Environment Variable `HF_TOKEN` chứa token HuggingFace của bạn vào Vercel Settings.
5. Deploy và copy URL API (ví dụ: `https://backend-api.vercel.app`).

**Cách 2: Deploy lên Kaggle (Dùng GPU)**
1. Upload file `backend/kaggle_notebook.ipynb` lên Kaggle.
2. Điền ngrok token và chạy Notebook để lấy Public URL.

### Bước 3: Deploy Frontend Next.js lên Vercel

1. Push toàn bộ source code lên GitHub.
2. Đăng nhập Vercel, chọn Import Project, trỏ vào thư mục `frontend`.
3. Trong phần cấu hình (Environment Variables), thêm biến `NEXT_PUBLIC_API_URL` và gán giá trị là URL Backend bạn vừa có ở Bước 2.
4. Bấm Deploy. Vercel sẽ tự động build Next.js và cung cấp đường link sống (Live URL) cho hệ thống của bạn!

---

*Tài liệu phân tích chuyên sâu: xem [`essays/CSDL_Misconception_Database_Final_Synced.docx`](essays/CSDL_Misconception_Database_Final_Synced.docx)*
