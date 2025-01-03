# 鼻子控制滑鼠系統

## 專案簡介
本專案利用 **OpenCV** 和 **MediaPipe** 技術，實現了一個基於面部特徵點的滑鼠控制系統。通過偵測使用者鼻子的移動，系統可以將鼻子的座標轉換為滑鼠的移動軌跡，達到使用鼻子控制滑鼠的效果。

未來將擴展專案功能，開發一個網頁系統，讓使用者不僅能透過鼻子控制滑鼠，還能在網頁上指定身體的哪個部位感到不適，從而提供更多互動與應用場景。

---

## 功能特色
- **實時面部追蹤：** 使用 MediaPipe 偵測面部特徵點，特別是鼻子的座標位置。
- **滑鼠移動控制：** 利用 PyAutoGUI 將鼻子座標映射到螢幕座標，模擬滑鼠移動。
- **高效影像處理：** 借助 OpenCV 處理攝影機影像，並進行座標轉換和畫面翻轉，使操作更直觀。
- **除錯工具：** 在影像中以圓點標記鼻子的位置，便於使用者觀察座標偵測結果。

---

## 使用技術
- **Python Libraries**
  - [OpenCV](https://opencv.org/)：用於影像處理和攝影機管理。
  - [MediaPipe](https://mediapipe.dev/)：用於面部網格特徵點的偵測。
  - [PyAutoGUI](https://pyautogui.readthedocs.io/)：用於模擬滑鼠的操作。
- **硬體需求**
  - 攝影機（用於捕捉面部影像）。
  - 螢幕解析度需支援 PyAutoGUI 的解析度讀取功能。

---

## 安裝與執行
### **環境需求**
- Python 3.7 或以上版本
- 安裝必要套件：
  ```bash
  pip install opencv-python mediapipe pyautogui



https://github.com/user-attachments/assets/6379b614-30bf-49ff-afb6-767aaa964e6c



