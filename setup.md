Để cài đặt thư viện Python trong dự án, cách phổ biến nhất là sử dụng pip thông qua terminal hoặc IDE. Bạn nên dùng môi trường ảo (virtual environment) để tránh xung đột thư viện giữa các dự án. Câu lệnh cơ bản là pip install <tên_thư_viện>. [1, 2, 3]
Các bước thực hiện chi tiết:

1.  Mở Terminal/Command Prompt: Mở cửa sổ dòng lệnh tại thư mục dự án của bạn.
2.  Tạo Môi trường ảo (Khuyên dùng):

- python -m venv venv
  - Kích hoạt: venv\Scripts\activate (Windows) hoặc source venv/bin/activate (macOS/Linux).

3.  Cài đặt thư viện:

- pip install <tên_thư_viện> (Ví dụ: pip install numpy).

4.  Lưu danh sách thư viện (Tùy chọn):

- pip freeze > requirements.txt để lưu lại các thư viện đã cài.

5.  Cài đặt từ file requirements:

- pip install -r requirements.txt. [1, 2, 3, 4, 5]

pip uninstall -r requirements.txt -y

Lưu ý: Nếu bạn dùng VS Code, hãy chọn interpreter (Ctrl+Shift+P -> Python: Select Interpreter) trỏ đúng vào thư mục venv vừa tạo. [1]
Nếu bạn muốn tôi hướng dẫn kỹ hơn, hãy cho biết:

- Bạn đang dùng hệ điều hành nào (Windows, macOS, hay Linux)?
- Bạn dùng IDE nào (VS Code, PyCharm, hay terminal)?
- Bạn có đang dùng Git không?

```shell
c:/Users/Phuc/PycharmProjects/Split_image_3/venv/Scripts/python.exe -m uvicorn server:app --reload


```
