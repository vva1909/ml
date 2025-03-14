import cv2

url = "http://your_video_stream_url_here"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()

    # Nếu không đọc được khung hình, thoát vòng lặp
    if not ret:
        print("Không thể nhận khung hình (stream end?). Thoát...")
        break

    # Hiển thị khung hình trong một cửa sổ
    cv2.imshow('Camera', frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()