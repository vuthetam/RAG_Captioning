# Repository Guidelines

## Environment Paths
- Git Bash: `C:\Program Files\Git\bin\bash.exe`
- Venv Python: `D:\workspace\ML\venv\Scripts\python.exe`

## Agent Behaviors
- **Confirmation Required**: Luôn phải hỏi ý kiến và chờ xác nhận của người dùng trước khi tiến hành viết, sửa đổi mã nguồn hoặc thực thi lệnh. Nếu người dùng chưa trả lời rõ ràng là đồng ý (ví dụ: "ok", "đồng ý", "triển khai đi"), phải hỏi lại, KHÔNG tự ý thực thi.
- **Lightweight Testing**: Khi chạy thử (test) các script nặng như `train`, `train_rag`, `build_kb`, `generate_captions`, AI Agent tự ngầm chạy kiểm tra giới hạn 1 batch (thông qua scratch script hoặc lệnh tạm thời) và bắt buộc ép `num_workers=0` (hoặc 1) cho Dataloader để tránh treo máy. TUYỆT ĐỐI KHÔNG thay đổi hay hardcode logic test (như `--test`, `head()`, `break`) vào mã nguồn chính thức.
- **Test Dataset**: Khi cần kiểm tra nhanh Dataset hoặc Dataloader, ưu tiên dùng `val_df`, `val_visual_features.h5` và `val_rag_contexts.parquet` thay vì tập train (tập train rất lớn, chậm load). Tập test không có cột `tokens` nên không dùng được cho các dataset có caption.
