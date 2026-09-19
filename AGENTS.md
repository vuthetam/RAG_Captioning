# Repository Guidelines

## Environment Paths
- Git Bash: `C:\Program Files\Git\bin\bash.exe`
- Venv Python: `D:\workspace\ML\venv\Scripts\python.exe`

## Agent Behaviors
- **Confirmation Required**: Luôn phải hỏi ý kiến và chờ xác nhận của người dùng trước khi tiến hành viết, sửa đổi mã nguồn hoặc thực thi lệnh.
- **Lightweight Testing**: Khi chạy thử (test) các script nặng như `train`, `build_kb`, `generate_captions`,... trên máy cục bộ, CHỈ được chạy giới hạn khoảng 10-100 sample (cắt slice dữ liệu) để tránh quá tải.
- **Lightweight Testing**: Khi chạy thử (test) các script nặng như `train`, `build_kb`, `generate_captions`, AI Agent tự ngầm chạy kiểm tra giới hạn 1 batch (thông qua scratch script hoặc lệnh tạm thời), TUYỆT ĐỐI KHÔNG thay đổi hay hardcode logic test (như `--test`, `head()`, `break`) vào mã nguồn chính thức.
