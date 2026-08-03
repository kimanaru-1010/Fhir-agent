# Knowledge Base — Symptom Reference (vector search, Qdrant)

Toàn bộ bệnh nằm gọn trong **một file JSON duy nhất**: `knowledge_base/diseases.json`
— một mảng (array) các bệnh, mỗi phần tử theo đúng schema bên dưới.
Muốn thêm bệnh mới: mở file này, thêm 1 object vào cuối mảng, KHÔNG cần sửa code.
`utils/knowledge_base.py` tự phát hiện khi file này bị sửa đổi (so sánh kích
thước + mtime) và **tự rebuild lại collection Qdrant** ở lần gọi tiếp theo.

## Cách hoạt động

- Vector store: **Qdrant**, mặc định chạy chế độ **local/embedded**
  (`QDRANT_MODE = "local"` trong `config/settings.py`) — lưu file lên đĩa tại
  `knowledge_base/.qdrant_index/`, không cần server riêng. Khi KB lớn dần lên
  hoặc cần nhiều worker process cùng truy cập (chế độ local chỉ cho 1 process
  tại một thời điểm), chỉ cần đổi `QDRANT_MODE = "server"` và trỏ `QDRANT_URL`
  vào một Qdrant server thật (`docker run -p 6333:6333 qdrant/qdrant`) —
  không cần đổi code.
- Embedding model lấy qua cùng server Ollama đang dùng cho LLM (xem
  `config/settings.py`: `EMBEDDING_URL` / `EMBEDDING_MODEL`, mặc định `bge-m3`).
  Cần chạy `ollama pull bge-m3` (hoặc model embedding khác hỗ trợ tiếng Việt)
  trước khi dùng.
- **Search chạy đúng 1 lần, ngay sau bước vision** — dùng `anamnesis` (lời khai
  ban đầu của bệnh nhân lúc upload ảnh) làm query, KHÔNG dùng `qa_history`
  (Q&A của 2 vòng phỏng vấn sau đó) và KHÔNG dùng `visual_observations` (mô tả
  hình ảnh của model vision — model vision nhiều khi mô tả sai/không chắc
  chắn, nếu dùng làm query thì sai số đó sẽ lan sang cả bước tra KB). Bệnh
  match được sẽ được thêm vào differential list *trước khi* planner sinh câu
  hỏi vòng 1, nên câu hỏi có thể phân biệt luôn cả các bệnh gợi ý từ KB.

## Schema

Mỗi bệnh chỉ gồm **3 trường**, không hơn — cả file là một mảng các object này:

```json
[
  {
    "disease": "Tên bệnh, tiếng Việt (tên sẽ được thêm vào differential list)",
    "disease_english": "Tên bệnh, tiếng Anh",
    "symptom_keywords": [
      "Dấu hiệu điển hình — ĐÂY LÀ PHẦN DUY NHẤT ĐƯỢC EMBED VÀ SEARCH",
      "vd: sẩn đỏ gồ cao, ranh giới rõ, ngứa nhiều về đêm"
    ]
  },
  {
    "disease": "Bệnh khác",
    "disease_english": "...",
    "symptom_keywords": ["..."]
  }
]
```

`disease` và `disease_english` chỉ dùng để hiển thị/định danh, KHÔNG được đưa
vào embedding. Chỉ `symptom_keywords` được nối lại và embed để search.

## Nguyên tắc viết `symptom_keywords`

- Chỉ mô tả những gì **nhìn thấy được bằng mắt** (màu sắc, hình dạng, ranh
  giới, bề mặt tổn thương...), **cảm nhận được khi sờ/chạm** (cứng, mềm, gồ,
  lõm...), **cảm giác của người bệnh** (ngứa, rát, đau, tê...), hoặc một
  **dấu hiệu đặc trưng/riêng biệt** (vd: dấu hiệu Auspitz, hạt Koplik, tổn
  thương hình bia bắn).
- KHÔNG mô tả diễn tiến theo thời gian (vd: "kéo dài nhiều tháng", "tái phát
  nhiều đợt", "lan rộng dần", "xuất hiện sau khi tiếp xúc X ngày") hay tiền
  sử/bệnh sử — những chi tiết này không phải là dấu hiệu quan sát/cảm nhận
  được ngay tại thời điểm khám.
- Viết như cách bệnh nhân/bác sĩ mô tả thật (embedding hiểu ngữ nghĩa, không
  cần trùng từ chính xác như kiểu match từ khóa cũ) — vd "cạo lớp vảy thấy
  chảy máu điểm nhỏ" tốt hơn chỉ ghi "có vảy".
- 4-8 mục chất lượng, đặc trưng riêng cho bệnh đó là đủ; không cần liệt kê
  hết mọi biến thể.

## Điều chỉnh độ nhạy

`match_kb_candidates(..., min_score=0.55)` trong `utils/knowledge_base.py` —
đây là **cosine similarity** (Qdrant trả về `score`, càng cao càng giống,
tối đa 1.0) — số càng cao thì càng chặt (ít false positive nhưng có thể bỏ
sót), số càng thấp thì càng lỏng. 0.55 là điểm khởi đầu hợp lý với `bge-m3`,
nên tinh chỉnh dựa trên vài ca thử thực tế.

## Mở rộng lên quy mô lớn

- Hiện tại mỗi lần KB thay đổi, `_ensure_index()` xoá và build lại toàn bộ
  collection — đơn giản, an toàn (xử lý đúng cả sửa/xoá bệnh), phù hợp khi KB
  còn ở quy mô vài chục-vài trăm bệnh.
- Nếu KB lên tới hàng nghìn bệnh, nên đổi sang **upsert tăng dần** (chỉ
  embed/ghi lại các entry đã thay đổi so với lần build trước, thay vì rebuild
  toàn bộ, và cân nhắc tách lại thành nhiều file nếu một file JSON quá lớn khó
  quản lý) — điểm chỉnh nằm trong `_ensure_index()`, phần "Full rebuild".