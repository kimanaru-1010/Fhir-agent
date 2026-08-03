# FHIR Agent - Healthcare Context Graph

FHIR Agent là ứng dụng trợ lý y tế dùng AI để truy vấn và tổng hợp dữ liệu FHIR
trên Neo4j, lưu hội thoại bằng PostgreSQL/pgvector, và có thêm workflow chẩn
đoán da liễu đa bước được tích hợp từ `skin-diagnostic-system`.

Ứng dụng hiện có hai chế độ chính trên frontend:

- `Chat`: hỏi đáp dữ liệu FHIR, hành trình chăm sóc, bệnh lý, thuốc, xét nghiệm,
  thanh toán, encounter, provider network.
- `Skin`: upload ảnh tổn thương da, nhập than phiền ban đầu, trả lời các câu hỏi
  lâm sàng PQRST, nhận kết quả chẩn đoán phân biệt.

## Kiến trúc

```text
fhir-agent/
├── backend/                  FastAPI backend
│   ├── app/
│   │   ├── agents/           FHIR agent, prompt, tool orchestration
│   │   ├── api/              FastAPI routers
│   │   ├── core/             Settings, constants, debug helpers
│   │   ├── db/               SQLAlchemy models/session
│   │   ├── dependencies/     Auth dependencies
│   │   ├── graph/            Neo4j/FHIR graph clients and models
│   │   ├── schemas/          API schemas
│   │   ├── services/         Chat, streaming, memory, auth services
│   │   └── skin_diagnostic/  Adapter/API/session store cho skin workflow
│   ├── pipeline/             Core pipeline gốc từ skin-diagnostic-system
│   ├── skills/               Prompt/skills gốc của skin workflow
│   ├── utils/                Helper gốc của skin workflow
│   ├── models/               Client/schema gốc của skin workflow
│   ├── config/               Settings gốc của skin workflow
│   ├── knowledge_base/       KB bệnh da liễu
│   └── tests/                Backend tests
├── frontend/                 Next.js + Chakra UI
├── cypher/                   Neo4j schema/index scripts
├── data/                     Ontology, fixture data
├── docker-compose.yml        Postgres, Qdrant, backend, frontend
├── Dockerfile.backend
├── Dockerfile.frontend
├── .env.example              Mẫu cấu hình
└── README.md
```

## Thành phần chạy

| Thành phần | Vai trò | Port mặc định |
| --- | --- | --- |
| Frontend | UI Chat/Skin | `3000` |
| Backend | FastAPI API | `8000` |
| PostgreSQL + pgvector | Auth DB, conversation memory, Mem0 | `5432` |
| Qdrant | Vector store cho skin knowledge base | `6333`, `6334` |
| Neo4j | FHIR graph chính | `7687`, browser thường là `7474` |

Lưu ý: `docker-compose.yml` hiện **không tạo Neo4j container**. Backend trong
Docker kết nối Neo4j qua `bolt://host.docker.internal:7687`, tức Neo4j cần chạy
sẵn trên máy host hoặc bạn chỉnh lại `NEO4J_URI`.

## Yêu cầu môi trường

- Docker Desktop
- Node.js 22 nếu chạy frontend local
- Python 3.10 đến dưới 3.14 nếu chạy backend local
- `uv` nếu chạy backend local
- Neo4j đang chạy và có dữ liệu FHIR
- Một OpenAI-compatible LLM endpoint cho chat agent
- Một OpenAI-compatible embedding endpoint cho pgvector/Mem0 và skin KB

## Cấu hình `.env`

Tạo file `.env` từ mẫu:

```powershell
Copy-Item .env.example .env
```

Hoặc trên bash:

```bash
cp .env.example .env
```

Các nhóm biến cần chỉnh:

### 1. Neo4j

Nếu chạy backend local:

```env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

Nếu chạy backend trong Docker Compose và Neo4j chạy trên máy host, compose đang
override thành:

```yaml
NEO4J_URI: bolt://host.docker.internal:7687
```

Nếu Neo4j chạy ở nơi khác, sửa `NEO4J_URI` trong `docker-compose.yml` hoặc đổi
compose để lấy trực tiếp từ `.env`.

### 2. LLM chính cho Chat/FHIR agent

`INTERNAL_LLM_BASE_URL` nên là base URL OpenAI-compatible có `/v1` nếu server của
bạn yêu cầu dạng đó.

```env
INTERNAL_LLM_BASE_URL=http://172.16.12.230:8000/v1
INTERNAL_LLM_API_KEY=internal-api-key
INTERNAL_LLM_MODEL=google/gemma-4-26B-A4B-it
```

### 3. Embedding chính cho Mem0/pgvector

```env
INTERNAL_EMBEDDING_BASE_URL=http://172.16.12.230:8004/v1
INTERNAL_EMBEDDING_API_KEY=internal-api-key
INTERNAL_EMBEDDING_MODEL=BAAI/bge-m3
INTERNAL_EMBEDDING_DIMS=1024
```

`INTERNAL_EMBEDDING_DIMS` phải khớp số chiều vector model embedding trả về.

### 4. PostgreSQL

Khi chạy bằng Docker Compose, backend được override để dùng host `postgres`.
Trong `.env` có thể để:

```env
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=fhir_agent
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/fhir_agent
```

Nếu chạy backend local nhưng Postgres vẫn chạy bằng Docker, dùng:

```env
POSTGRES_HOST=localhost
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/fhir_agent
```

### 5. JWT/Auth

```env
JWT_SECRET_KEY=replace-with-a-long-random-secret
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
```

Frontend yêu cầu đăng ký/đăng nhập trước khi dùng Chat hoặc Skin.

### 6. Skin diagnostic

Project đang giữ cả biến `SKIN_*` của adapter và biến gốc từ
`skin-diagnostic-system`.

Với core pipeline gốc, các biến quan trọng là:

```env
VISION_MODEL_URL=http://172.16.12.230:8000
VISION_MODEL_DEFAULT=gemma-4-26B-A4B-it
VISION_MODEL_PREFER=gemma-4
VISION_MODEL_TIMEOUT=600

REASONING_MODEL_URL=http://172.16.12.230:8000
REASONING_MODEL_DEFAULT=gemma-4-26B-A4B-it
REASONING_MODEL_PREFER=gemma-4
REASONING_MODEL_TIMEOUT=1200

EMBEDDING_URL=http://172.16.12.230:8004
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_TIMEOUT=120

QDRANT_MODE=server
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=disease_symptoms
```

Với các biến `VISION_MODEL_URL`, `REASONING_MODEL_URL`, `EMBEDDING_URL`, dùng URL
gốc **không thêm `/v1`**, vì code gốc tự gọi `/v1/chat/completions`,
`/v1/models`, và `/v1/embeddings`.

Khi chạy bằng Docker Compose, backend tự dùng:

```env
QDRANT_URL=http://qdrant:6333
```

để gọi Qdrant container.

## Chạy bằng Docker Compose

Đây là cách khuyến nghị cho project hiện tại.

1. Đảm bảo Neo4j đang chạy trên máy host và lắng nghe port `7687`.
2. Chỉnh `.env`.
3. Build và chạy:

```powershell
docker compose up --build -d
```

Kiểm tra container:

```powershell
docker compose ps
```

Kiểm tra backend:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Hoặc:

```bash
curl http://localhost:8000/health
```

Mở UI:

```text
http://localhost:3000
```

Backend có bước warm-up skin knowledge base khi startup. Nếu health chưa phản hồi
ngay sau restart, đợi khoảng 1-2 phút rồi thử lại.

## Chạy local development

Cách này hữu ích khi muốn debug backend/frontend trực tiếp.

### 1. Chạy hạ tầng phụ bằng Docker

```powershell
docker compose up -d postgres qdrant
```

Neo4j vẫn cần chạy riêng.

### 2. Backend local

```powershell
cd backend
uv sync --extra dev
$env:POSTGRES_HOST="localhost"
uv run uvicorn app.main:app --reload --port 8000
```

Nếu dùng bash:

```bash
cd backend
uv sync --extra dev
POSTGRES_HOST=localhost uv run uvicorn app.main:app --reload --port 8000
```

### 3. Frontend local

```powershell
cd frontend
npm install
npm run dev
```

Frontend mặc định gọi API qua:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

Nếu cần đổi backend URL, set biến này trước khi chạy frontend.

## Khởi tạo dữ liệu FHIR/Neo4j

Các script dữ liệu nằm trong `backend/scripts` và `data`.

Chạy seed:

```powershell
cd backend
uv run python scripts/generate_data.py
```

Script này dùng thông tin Neo4j trong `.env`.

## Kiểm thử

Backend:

```powershell
cd backend
$env:POSTGRES_HOST="localhost"
python -m pytest
```

Frontend build:

```powershell
cd frontend
npm run build
```

Frontend e2e nếu cần:

```powershell
cd frontend
npx playwright install
npm run test:e2e
```

## Các API chính

Backend API prefix là `/api`.

Auth:

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/users/me`

Chat/FHIR:

- `GET /api/conversations`
- `POST /api/conversations`
- `GET /api/conversations/{id}/messages`
- `POST /api/conversations/{id}/messages`
- Streaming endpoints nằm trong routers frontend đang dùng.

Skin diagnostic:

- `POST /api/skin-diagnostics/start`
- `GET /api/skin-diagnostics/{run_id}/status`
- `POST /api/skin-diagnostics/{run_id}/answers`
- `GET /api/skin-diagnostics/uploads/{filename}`
- `DELETE /api/skin-diagnostics/{run_id}`

## Luồng Skin Diagnostic

Pipeline skin hiện chạy theo core gốc:

1. `visual_extract`: phân tích ảnh bằng vision model.
2. Knowledge base augmentation: truy vấn Qdrant để bổ sung differential.
3. `clinical_planner_round1`: sinh 5 câu hỏi PQRST vòng 1.
4. `user_interview_round1`: frontend gửi câu trả lời Yes/No.
5. `clinical_planner_round2`: sinh 5 câu hỏi mới dựa trên vòng 1.
6. `user_interview_round2`: frontend gửi câu trả lời Yes/No.
7. `diagnostic_reasoning`: trả kết quả chẩn đoán phân biệt và reasoning.

Kết quả là decision support, không phải chẩn đoán y khoa cuối cùng.

## Lưu ý vận hành

- Backend Docker mount `./backend:/app`, nên thay đổi code backend có thể làm
  Uvicorn reload.
- Frontend Docker mount `./frontend:/app`, nên thay đổi frontend có thể hot reload.
- Qdrant chạy trong Docker Compose.
- Postgres chạy trong Docker Compose.
- Neo4j không nằm trong Compose hiện tại.
- Skin session snapshot và upload nằm dưới
  `backend/app/skin_diagnostic/data/`.
- Không commit `.env`, session uploads, hoặc dữ liệu nhạy cảm.

## Troubleshooting

### Backend health chưa phản hồi sau khi restart

Backend warm-up Neo4j, Mem0 và skin KB. Đợi 1-2 phút rồi gọi lại:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Xem log:

```powershell
docker compose logs --tail 120 backend
```

### Neo4j unavailable

- Kiểm tra Neo4j đang chạy.
- Kiểm tra port `7687`.
- Nếu backend chạy trong Docker, dùng `host.docker.internal`.
- Nếu backend chạy local, dùng `localhost`.

### LLM trả lỗi model not found

- Kiểm tra endpoint `/v1/models` của LLM server.
- Với `INTERNAL_LLM_BASE_URL`, thường dùng URL có `/v1`.
- Với skin upstream vars `VISION_MODEL_URL` và `REASONING_MODEL_URL`, dùng URL
  không có `/v1`.
- Nếu server expose model dạng `google/gemma-...`, dùng đúng model id hoặc chỉnh
  `*_PREFER` để resolver chọn model phù hợp.

### Qdrant không kết nối được

Khi chạy Docker Compose, backend phải dùng:

```env
QDRANT_URL=http://qdrant:6333
```

Khi chạy backend local:

```env
QDRANT_URL=http://localhost:6333
```

### Frontend báo chưa đăng nhập

Đăng ký hoặc đăng nhập ở tab Chat trước. Token được lưu trong `localStorage`.

### Muốn reset containers

Không xóa volume nếu còn cần dữ liệu:

```powershell
docker compose down
docker compose up --build -d
```

Nếu muốn xóa cả Postgres/Qdrant data:

```powershell
docker compose down -v
docker compose up --build -d
```

## Ghi chú phát triển

- Backend app root chỉ giữ `main.py`; code tính năng nên nằm trong package con.
- Prompt chính của FHIR agent nằm trong `backend/app/agents/fhir.py`.
- Prompt/skills của skin workflow nằm trong `backend/skills`.
- Core skin pipeline/vendor từ repo gốc nằm trong `backend/pipeline`, `backend/utils`,
  `backend/models`, `backend/config`, `backend/tools`, `backend/knowledge_base`.
- Khi thay đổi prompt hoặc pipeline, chạy ít nhất:

```powershell
cd backend
python -m pytest tests/test_agent.py tests/test_skin_diagnostic.py
```

