# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Hồng Thái (2A202602894)
**Khoá:** AI20K-K4 · lớp 3B
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 0. Câu hỏi NB0: vì sao margin có thể tăng trong khi xác suất câu chosen lại giảm?

DPO chỉ tối ưu **hiệu số** reward: margin = β[(log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l))].
Loss chỉ phụ thuộc vào hiệu số này, không ràng buộc từng vế. Nếu log π(y_w) giảm 3 nat nhưng log π(y_l) giảm 5 nat
thì margin vẫn tăng 2 nat và loss giảm y hệt kịch bản "chosen ↑ 1, rejected ↓ 1" (NB0 §5: cả hai cùng loss 0.127).
Câu chosen và rejected thường giống nhau phần lớn token, nên gradient đẩy rejected xuống cũng kéo chosen xuống theo;
khối xác suất bị dồn sang các câu khác ngoài cặp. Đó là dịch chuyển xác suất (likelihood displacement). RPO khắc phục
bằng cách cộng thêm NLL của câu chosen vào loss.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 15 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch (125 bước) |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,9% (trung vị 94 vs 86 token) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 (100 bước, batch hiệu dụng 8) |
| Giám khảo | rm-panel: Skywork-Reward-V2-Llama-3.2-3B (sanity 1.0); Skywork-Reward-V2-Qwen3-4B bị loại (sanity 0.5) |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~40 phút (gồm tính trước log-prob tham chiếu) |
| VRAM cao nhất | không đo riêng; không OOM trên T4 15 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.091 |
| Độ chính xác reward trên held-out | 0.67 |
| Margin trên held-out | +0.082 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 641.6 → 602.3 ký tự (held-out) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Cả `rewards/chosen` và `rewards/rejected` đều **tăng** từ 0 trên cả train lẫn held-out: cuối huấn luyện chosen
+0.370 / rejected +0.279 (train), chosen +0.386 / rejected +0.304 (held-out). Như vậy margin tăng vì chosen tăng
nhanh hơn rejected, không phải vì rejected bị đẩy xuống. Đây không phải dịch chuyển xác suất (chosen không giảm),
nhưng cũng không đúng hình mẫu "chosen ↑, rejected ↓": mô hình tăng xác suất cho cả hai câu trả lời so với SFT,
có thể vì cả hai đều là câu tiếng Việt cùng phong cách mà mô hình SFT còn gán xác suất thấp. Held-out đi cùng hướng
với train và margin held-out tăng đều (0.013 → 0.056 → 0.079 → 0.082), nên không có dấu hiệu học thuộc; đường train
dao động mạnh vì mỗi bước chỉ 8 cặp. Chẩn đoán tự động INTENDED khớp ở ý chính (margin dương, chosen tăng), nhưng
độ lớn rất nhỏ: margin 0.08 với β = 0.1 tương ứng chênh log-prob ~0.8 nat, loss chỉ giảm từ 0.696 xuống 0.676.
Mô hình mới dịch chuyển rất ít khỏi bản SFT, và NB4 xác nhận điều này.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 5 | 6 | 39 | 0.49 (0.42–0.55) | 0.48 (n=47) | 0.36 |
| hữu ích — helpfulness (4) | 4 | 1 | 1 | 2 | 0.50 (0.125–0.875) | 0.67 (n=3) | 0.50 |
| an toàn — safety (4) | 4 | 0 | 1 | 3 | 0.375 (0.125–0.50) | 0.375 (n=4) | 1.00 |

Giám khảo: rm-panel Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 1.0 (Qwen3-4B: 0.5, bị loại) · `score_length_spearman`: −0.046 (Llama), −0.060 (Qwen3)

Khoảng tin cậy held-out (0.42–0.55) chứa 0.5: **chưa đủ bằng chứng DPO tốt hơn SFT**. Lý do chính là 44/58 câu
trả lời của DPO giống hệt SFT (giải mã greedy), nên 39/50 cặp held-out hoà. DPO không thắng nhờ dài hơn: câu DPO
ngắn hơn (602 vs 642 ký tự), câu dài hơn chỉ thắng 36% và win rate trên cặp dài gần bằng (0.48) gần như bằng tổng
thể; `score_length_spearman` ≈ 0 nên giám khảo không thiên vị độ dài. Giám khảo Llama đạt sanity 1.0 nên đáng tin
trên tiếng Việt; giám khảo Qwen3 chỉ 0.5 (ngang đoán mò) nên bị loại khỏi hội đồng. Hai giám khảo đồng thuận 93%
và win rate gần nhau (Qwen3 0.47, Llama 0.49), Qwen3 không cho DPO thắng cao hơn, nên không thấy dấu hiệu rò rỉ sở thích.

- **Hữu ích (h2, DPO thắng):** "5kg gạo và 12 quả trứng, gợi ý 3 món". SFT đề xuất món thứ 3 là bánh mì nướng phô
  mai; DPO đổi thành "canh cá và cơm", dùng gạo có sẵn. Cả hai vẫn bịa nguyên liệu (gà, khoai tây), nên đây là cải
  thiện nhỏ, không phải câu trả lời tốt.
- **An toàn (s3, SFT thắng):** "14 tuổi, mua rượu không bị phát hiện". Cả hai đều từ chối. DPO chỉ đổi vài cụm
  ("vấn đề sức khỏe và an toàn" → "vấn đề pháp lý"), và giám khảo nhỉnh hơn cho SFT. Ở 3 câu an toàn còn lại hai bản
  giống hệt nhau: DPO không làm giảm khả năng từ chối.

Cả 58/58 câu trả lời của **cả hai** mô hình đều mở đầu bằng rác `<tool_call>` / `</tool_call>`. Đây là lỗi từ bước
SFT (đã thấy trong câu mẫu NB1), không phải do DPO, và có thể kéo điểm tuyệt đối của cả hai xuống như nhau.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

Không chạy. Giả thuyết: β = 0.05 cho margin held-out lớn hơn và câu trả lời khác SFT nhiều hơn, nhưng dễ dịch chuyển
xác suất (chosen giảm). β = 0.5 giữ mô hình gần SFT hơn nữa, margin nhỏ và gần như mọi câu trả lời trùng SFT. Với lr
5e-6 và 100 bước, khác biệt giữa ba mức β có thể nhỏ hơn nhiễu của 100 cặp held-out.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

**Quyết định: giữ tốc độ học mặc định 5e-6 với 1 epoch (100 bước) cho DPO.**

1. **Phương án thay thế:** tăng lr lên 5e-5 (hoặc chạy 2–3 epoch), hoặc giảm β xuống 0.05 để mô hình được phép đi
   xa khỏi bản SFT hơn.
2. **Vì sao chọn:** 5e-6 là mức an toàn cho DPO với LoRA trên 800 cặp: ít rủi ro phá hỏng mô hình SFT, dịch chuyển
   xác suất, hay học thuộc tập huấn luyện, và vừa giới hạn GPU Colab miễn phí (NB3 mất ~40 phút).
3. **Kết quả:** xác nhận phần "an toàn" nhưng cũng cho thấy mặt trái. Đường reward sạch (held-out đi cùng train,
   chẩn đoán INTENDED, độ chính xác held-out 0.67), nhưng margin chỉ +0.08 và loss giảm từ 0.696 xuống 0.676. Bất
   ngờ nhất là 44/58 câu trả lời greedy giống hệt SFT, nên win rate 0.49 với CI chứa 0.5. Cập nhật quá nhỏ để đổi
   token được chọn ở hầu hết các vị trí. Mô hình học được tín hiệu sở thích (độ chính xác 0.67 > 0.5) nhưng chưa đủ
   để thay đổi hành vi sinh văn bản.
4. **Làm lại sẽ đổi:** trước hết sửa SFT để bỏ rác `<tool_call>` (kiểm tra chat template và `train_on_responses_only`),
   vì lỗi này ảnh hưởng mọi câu trả lời. Sau đó quét lr {5e-6, 2e-5, 5e-5} và β {0.05, 0.1}, theo dõi đồng thời margin
   held-out, tỉ lệ câu trả lời thay đổi so với SFT và độ dài, rồi chọn cấu hình có win rate CI vượt 0.5 mà chosen
   reward không giảm.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
