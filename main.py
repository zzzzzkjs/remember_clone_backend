import sys
import matplotlib.pyplot as plt
import numpy as np
from pytesseract import pytesseract
from PIL import Image
import cv2

def reorderPts(pts):
    idx = np.lexsort((pts[:, 1], pts[:, 0]))  # 칼럼0 -> 칼럼1 순으로 정렬한 인덱스를 반환
    pts = pts[idx]  # x좌표로 정렬

    if pts[0, 1] > pts[1, 1]:
        pts[[0, 1]] = pts[[1, 0]]

    if pts[2, 1] < pts[3, 1]:
        pts[[2, 3]] = pts[[3, 2]]

    return pts

# cv_image = cv2.imread("remember_clone_backend/resource/my_small.jpg")
# cv_image = cv2.imread("remember_clone_backend/resource/my.jpg")
# cv_image = cv2.imread("remember_clone_backend/resource/my2.jpg")
# cv_image = cv2.imread("remember_clone_backend/resource/test3.jpg")
cv_image = cv2.imread("remember_clone_backend/resource/test4.jpg")
if cv_image is None:
    print('image load is failed')
    sys.exit()

print("cv_image type : ", type(cv_image))

cv_image = cv2.resize(cv_image, (0, 0), fx=0.3, fy=0.3)

# cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

cv_image_pre = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

# 이미지 노이즈 제거를 위한 Blur 처리
cv_image_pre = cv2.GaussianBlur(cv_image_pre, (5,5), 0)

# 엣지 추출을 위해 Canny 함수 사용
# cv_image_pre = cv2.Canny(cv_image_pre, 100,200)

th, cv_image_pre = cv2.threshold(cv_image_pre, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 윤곽선 추출 (명함 테두리 검출용)
contours, _ = cv2.findContours(cv_image_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

maxArea = 0
outLine = None

for cnt in contours:
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # 사각형이 아닐경우 무시
    if not cv2.isContourConvex(approx) or len(approx) != 4:
        continue
    # 가장 큰 영역의 윤곽선을 명함 테두리로 추출
    if cv2.contourArea(approx) > maxArea:
        maxArea = cv2.contourArea(approx)
        outLine = approx

cv2.drawContours(cv_image,[outLine],0,(0,0,255),2)

w, h = 900, 500

# 시계방향 정렬
srcQuad = reorderPts(outLine.reshape(4,2).astype(np.float32))
dstQuad = np.array([[0, 0], [0, h], [w, h], [w, 0]], np.float32)

# srcQuad = np.array([[hull[0, 0, :]], [hull[1, 0, :]], [hull[2, 0, :]], [hull[3, 0, :]]]).astype(np.float32)
# dstQuad = np.array([[w,0], [0,0], [0,h], [w,h] ]).astype(np.float32)

pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
cv_image_pre = cv2.warpPerspective(cv_image, pers, (w, h))

# 글자 영역을 뽑아내기위한 팽창연산 용 커널, 대부분 가로방향(죄->우)으로 텍스트가 나열되어있기 때문에 kernel을 3,5로 잡음
# 5,5로 잡을 시 세로간격이 좁은 부분이 하나의 영역으로 합쳐짐
# kernel = np.ones((3, 5), np.uint8)
# 팽창연산을 통해 가까운 글자 영역을 합침
# cv_image_pre = cv2.dilate(cv_image_pre, kernel, iterations = 1)

# 윤곽선 추출 (글자 영역 검출용)
# contours, _ = cv2.findContours(cv_image_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cv_image_pre = np.expand_dims(cv_image_pre, axis=2).repeat(3,axis=2)

# for cnt in contours:
#     epsilon = 0.02 * cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, epsilon, True)
#     x, y, w, h = rectangle = cv2.boundingRect(approx)
#     # 추출된 사각형 영역이 일정 크기 이상일 경우 텍스트 영역으로 판단
#     if w*h > 200:
#         cv2.rectangle(cv_image_pre, (x,y), (x+w, y+h), (0,255,0), 1)

# print("outLine : ", outLine)
# x, y, w, h = outLine
# cv2.rectangle(cv_image, (x,y), (x+w, y+h), (0,255,0), 1)

cv_image_pre = cv2.cvtColor(cv_image_pre, cv2.COLOR_BGR2RGB)

text = pytesseract.image_to_string(cv_image_pre, lang='Hangul+eng')
print(text)
# with open("remember_clone_backend/resource/sample.txt", "w") as f:
#     f.write(text)

cv2.imshow("test2", cv_image)
cv2.imshow("dst", cv_image_pre)
cv2.waitKey(0)
cv2.destroyAllWindows()


