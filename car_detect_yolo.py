from imutils.video import FPS
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import argparse # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import imutils # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import time # 시간 처리 모듈
import cv2 # opencv 모듈
import os # 운영체제 기능 모듈

# YOLO 모델이 학습된 coco 클래스 레이블
labelsPath = "D://data//coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
# YOLO 가중치 및 모델 구성에 대한 경로
weightsPath = "D://data//yolov3.weights"  #os.path.sep.join(["yolo-coco", "yolov3.weights"]) # 가중치
configPath = "D://data//yolov3.cfg" #os.path.sep.join(["yolo-coco", "yolov3.cfg"]) # 모델 구성
# COCO 데이터 세트(80 개 클래스)에서 훈련된 YOLO 객체 감지기 load
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# YOLO에서 필요한 output 레이어 이름
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
vs = cv2.VideoCapture('d://data//car1.mp4')    

# fps 정보 초기화
fps = FPS().start()

writer = None
(W, H) = (None, None)

# 비디오 스트림 프레임 반복
while True:
    # 프레임 읽기
    ret, frame = vs.read()

    # 프레임 크기 지정
    frame = imutils.resize(frame, width=600)

    # 프레임 크기
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (256,256), swapRB=True, crop=False)
    
    # 객체 인식
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # bounding box, 확률 및 클래스 ID 목록 초기화
    boxes = []
    confidences = []
    classIDs = []
    
    # layerOutputs 반복
    for output in layerOutputs:
        # 각 클래스 레이블마다 인식된 객체 수 만큼 반복
        for detection in output:
            # 인식된 객체의 클래스 ID 및 확률 추출
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            # 객체 확률이 최소 확률보다 큰 경우
            if confidence > 0.5:
                # bounding box 위치 계산
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int") # (중심 좌표 X, 중심 좌표 Y, 너비(가로), 높이(세로))
                
                # bounding box 왼쪽 위 좌표
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                # bounding box, 확률 및 클래스 ID 목록 추가
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    # bounding box가 겹치는 것을 방지(임계값 적용)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    
    # 인식된 객체가 있는 경우
    if len(idxs) > 0:
        # 모든 인식된 객체 수 만큼 반복
        for i in idxs.flatten():
            # bounding box 좌표 추출
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # random으로 지정된 색깔
            color = [int(c) for c in COLORS[classIDs[i]]]
            
            # bounding box 출력
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            
            # 클래스 ID 및 확률
            text = "{} : {:.2f}%".format(LABELS[classIDs[i]], confidences[i])
            
            # label text 잘림 방지
            y = y - 15 if y - 15 > 15 else y + 15
            
            # text 출력
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 프레임 출력
    cv2.imshow("Real-Time Object Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # 'q' 키를 입력하면 종료
    if key == ord("q"):
        break    
    # fps 정보 업데이트
    fps.update()    

# fps 정지 및 정보 출력
fps.stop()
print("[재생 시간 : {:.2f}초]".format(fps.elapsed()))
print("[FPS : {:.2f}]".format(fps.fps()))

# 종료
vs.release()
cv2.destroyAllWindows()