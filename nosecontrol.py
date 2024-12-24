import cv2
import mediapipe as mp
import pyautogui

# 初始化 Mediapipe 的偵測器
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化攝影機
cap = cv2.VideoCapture(0)

# 獲取螢幕的解析度
screen_width, screen_height = pyautogui.size()

# 設定攝影機畫面大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 開始循環
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 翻轉畫面，讓操作更直覺
    frame = cv2.flip(frame, 1)

    # 將影像轉為 RGB 格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 進行偵測
    result = face_mesh.process(rgb_frame)

    # 確保偵測到面部網格
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Mediapipe 的鼻子座標為第 1 個點 (index 1)
            nose = face_landmarks.landmark[1]

            # 取得鼻子座標的比例位置
            x = int(nose.x * frame.shape[1])
            y = int(nose.y * frame.shape[0])

            # 將鼻子座標轉換到螢幕座標
            screen_x = int(nose.x * screen_width)
            screen_y = int(nose.y * screen_height)

            # 使用 PyAutoGUI 移動滑鼠
            pyautogui.moveTo(screen_x, screen_y)

            # 在畫面上繪製鼻子座標 (用於除錯)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # 顯示影像
    cv2.imshow('Nose Mouse Control', frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()