img = cv2.imread('/home/intel/사진/images.jpeg', cv2.IMREAD_GRAYSCALE)  # 이미지를 그레이스케일로 읽기
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  # 엣지 검출을 위한 커널
print(kernel)

output = cv2.filter2D(img, -1, kernel)  # 필터 적용
cv2.imshow('edge', output)  # 결과 이미지 출력
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 창 닫기
