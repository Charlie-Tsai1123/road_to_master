from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import torch
import cv2
import numpy as np

# 清空 GPU 緩存
torch.cuda.empty_cache()

# 加載 YOLO 人臉偵測模型
face_detector = YOLO('./yolov8n.pt')

# 加載 InceptionResnetV1 人臉識別模型
face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()

# 圖像處理函數
def process_face(face_img):
    face_img = cv2.resize(face_img, (160, 160))  # 調整到模型輸入大小
    face_img = np.transpose(face_img, (2, 0, 1))  # 調整通道順序 (C, H, W)
    face_img = face_img / 255.0  # 將像素值歸一化到 [0, 1] 範圍
    face_img = torch.tensor(face_img).float().unsqueeze(0)  # 轉為 tensor 並增加 batch 維度
    return face_img

# 特徵提取函數
def get_embedding(face_img, model):
    face_tensor = process_face(face_img)
    with torch.no_grad():  # 停用梯度計算（加速並節省內存）
        embedding = model(face_tensor)
    return embedding.squeeze().detach().numpy()  # 確保輸出為一維向量

# 加載參考圖片並提取嵌入向量
reference_image_path_1 = 'person1.jpg'
reference_image_path_2 = 'person2.jpg'

reference_image_1 = cv2.imread(reference_image_path_1)
reference_image_2 = cv2.imread(reference_image_path_2)

# 使用 YOLO 偵測人臉並提取參考嵌入向量
def get_reference_embedding(reference_image, reference_image_path):
    reference_results = face_detector(reference_image)
    if reference_results[0].boxes:
        # 獲取 YOLO 偵測的邊界框座標
        x1, y1, x2, y2 = map(int, reference_results[0].boxes[0].xyxy[0].tolist())
        reference_face = reference_image[y1:y2, x1:x2]  # 擷取人臉區域
        reference_embedding = get_embedding(reference_face, face_recognizer)
        return reference_embedding
    else:
        print(f"未在參考圖片 {reference_image_path} 中檢測到人臉")
        return None

# 提取兩個參考人的嵌入向量
reference_embedding_1 = get_reference_embedding(reference_image_1, reference_image_path_1)
reference_embedding_2 = get_reference_embedding(reference_image_2, reference_image_path_2)

# 開啟攝像頭
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 從攝像頭讀取一幀
    ret, frame = cap.read()
    if not ret:
        print("無法從攝像頭讀取影像")
        break

    # 使用 YOLO 模型偵測人臉
    results = face_detector(frame)

    for box in results[0].boxes:
        # 獲取邊界框座標
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        face_img = frame[y1:y2, x1:x2]  # 擷取人臉區域
        confidence = box.conf[0]

        # 取得人臉的嵌入向量
        detected_embedding = get_embedding(face_img, face_recognizer)

        # 計算與兩個參考嵌入向量的餘弦相似度
        cosine_similarity_1 = np.dot(reference_embedding_1, detected_embedding) / (
            np.linalg.norm(reference_embedding_1) * np.linalg.norm(detected_embedding)
        )
        cosine_similarity_2 = np.dot(reference_embedding_2, detected_embedding) / (
            np.linalg.norm(reference_embedding_2) * np.linalg.norm(detected_embedding)
        )

        # 根據相似度決定是哪一個人
        if cosine_similarity_1 > cosine_similarity_2:
            label = f"Person 1 | Similarity: {cosine_similarity_1:.2f} | Conf: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 用綠色框標註
            text_color = (0, 255, 0)  # 綠色文字
        else:
            label = f"Person 2 | Similarity: {cosine_similarity_2:.2f} | Conf: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 用紅色框標註
            text_color = (0, 0, 255)  # 紅色文字

        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # 顯示當前幀
    cv2.imshow('Detected Image', frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭並關閉視窗
cap.release()
cv2.destroyAllWindows()