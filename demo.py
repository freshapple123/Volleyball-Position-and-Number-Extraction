import cv2
from ultralytics import YOLO
import ultralytics.utils as ultralytics_utils
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# 특정 조건에 따라 점을 필터링하는 함수 예시
def is_within_court_area(x, y, court_points):
    # 첫 번째, 세 번째, 네 번째 점의 좌표 참조
    first_y = court_points[0][1]
    third_x, third_y = court_points[2]
    fourth_x = court_points[3][0]

    # 점이 주어진 경계 안에 있는지 확인하는 조건문
    if (first_y < y < third_y) and (fourth_x < x < third_x):
        return True
    return False


# 불필요한 메시지를 억제하기 위해 Ultralytics 로그 수준을 WARN으로 설정
ultralytics_utils.LOGGER.setLevel(logging.WARN)

# YOLOv8 Pose 모델 로드
model = YOLO("yolo11x-pose.pt")

# 비디오 파일 또는 웹캠 스트림 열기
video_path = "C:/Users/PC/Desktop/배구/top_short.mp4"
cap = cv2.VideoCapture(video_path)

# 마우스 클릭으로 선택한 배구 코트의 점을 저장할 리스트
court_points = []
start_video = False  # 'w' 키가 눌리기 전까지 비디오 재생을 일시 정지하는 플래그


# 마우스 클릭 이벤트 콜백 함수 정의
def select_point(event, x, y, flags, param):
    global court_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭한 좌표 저장 (최대 4점)
        if len(court_points) < 4:
            court_points.append((x, y))
            print(f"선택된 점: {x}, {y}")
        else:
            print("이미 4점을 선택했습니다.")


# 첫 번째 프레임을 읽고 점 선택을 위한 마우스 클릭 설정
ret, first_frame = cap.read()

# 첫 번째 프레임이 제대로 로드되었는지 확인
if not ret:
    print("첫 번째 프레임을 로드할 수 없습니다.")
    cap.release()
    exit()

# 점 선택을 허용하기 위해 마우스 콜백 설정
cv2.namedWindow("Set Court Points")
cv2.setMouseCallback("Set Court Points", select_point)

while True:
    # 4점이 선택되었을 때 선을 그림
    if len(court_points) == 4:
        for i in range(4):
            cv2.line(
                first_frame,
                court_points[i],
                court_points[(i + 1) % 4],
                (0, 255, 0),
                2,
            )

    # 첫 번째 프레임 표시
    cv2.imshow("Set Court Points", first_frame)

    # 'w' 키를 눌러 비디오 재생을 시작하거나 'q' 키로 종료 대기
    key = cv2.waitKey(1) & 0xFF
    if key == ord("w") and len(court_points) == 4:
        start_video = True  # 비디오 재생 시작
        break
    elif key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Set Court Points")  # 점 선택 창 닫기

# 비디오 스트림 재생 루프
if start_video:
    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()

            # 프레임이 제대로 로드되었는지 확인
            if not ret:
                print("프레임을 읽을 수 없습니다. 비디오가 끝났을 수 있습니다.")
                break

            # 프레임 시각화
            annotated_frame = frame.copy()  # 주석 처리를 위한 복사본 생성

            # 선택한 코트 포인트를 프레임에 그리기
            if len(court_points) == 4:
                for i in range(4):
                    cv2.line(
                        annotated_frame,
                        court_points[i],
                        court_points[(i + 1) % 4],
                        (0, 255, 0),
                        2,
                    )

            # 실시간 프레임 표시
            cv2.imshow("YOLOv8 Pose Detection", annotated_frame)

        # 키 입력 처리
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            paused = not paused  # 일시 정지 상태 전환
        elif key == ord("e") and paused:  # 'e'가 눌리면 일시 정지 상태에서
            print("일시 정지 상태에서 'e'가 눌렸습니다.")
            # 현재 프레임에서 YOLO 포즈 감지 실행
            frame_custom = frame
            last_frame = frame
            results = model(frame)
            # 주석 처리된 프레임에서 결과 시각화
            annotated_frame = results[0].plot()

            # 사람의 바운딩 박스 정보를 저장할 리스트
            person_boxes = []

            # 감지된 객체들 중에서 사람 객체만 선택하여 바운딩 박스 좌표 저장
            for box in results[0].boxes:
                class_id = box.cls.item()  # 클래스 ID 추출
                class_name = model.names[class_id]  # 클래스 이름 가져옴

                # 클래스 이름이 'person'인 경우에만 바운딩 박스 좌표를 저장
                if class_name == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2))  # 좌표를 리스트에 추가

            # 선택한 코트 포인트 다시 그리기
            if len(court_points) == 4:
                for i in range(4):
                    cv2.line(
                        annotated_frame,
                        court_points[i],
                        court_points[(i + 1) % 4],
                        (0, 255, 0),
                        2,
                    )

            # 포즈 감지 결과가 있는 프레임 표시
            cv2.imshow("YOLOv8 Pose Detection", annotated_frame)

            # 저장된 사람의 바운딩 박스 위치 확인
            for i, box in enumerate(person_boxes):
                print(f"Person {i + 1} Bounding Box: {box}")

            key = cv2.waitKey(0) & 0xFF  # 키 입력 대기
            if key == ord("r"):  # 'r' 키가 눌렸는지 확인
                print("'r' 키가 눌렸습니다.")
                # 포즈 감지 실행하고 키포인트 얻기
                right_foot_index = 16  # 오른발의 인덱스
                left_foot_index = 15  # 왼발의 인덱스
                keypoints = results[0].keypoints.xyn.cpu().numpy()

                # 오른발의 좌표 추출
                right_foot = keypoints[:, right_foot_index, :2]  # x, y 좌표

                # 왼발의 좌표 추출
                left_foot = keypoints[:, left_foot_index, :2]  # x, y 좌표

                # 이미지 크기 가져오기
                height, width, _ = frame.shape  # 프레임을 사용

                # 실제 픽셀 좌표로 변환
                right_foot_pixel = right_foot * np.array([width, height])
                left_foot_pixel = left_foot * np.array([width, height])

                # 코트 영역 내 발이 있는 사람만 남기기 위한 필터링
                filtered_person_boxes = []

                # 각 사람의 바운딩 박스에 대해 발 위치를 확인
                for i, (x1, y1, x2, y2) in enumerate(person_boxes):
                    # 현재 사람의 오른발과 왼발 좌표 가져오기
                    right_foot_pos = right_foot_pixel[i]
                    left_foot_pos = left_foot_pixel[i]

                    # 오른발 또는 왼발이 코트 영역 내에 있는지 확인
                    if is_within_court_area(
                        right_foot_pos[0], right_foot_pos[1], court_points
                    ) or is_within_court_area(
                        left_foot_pos[0], left_foot_pos[1], court_points
                    ):
                        filtered_person_boxes.append(
                            (x1, y1, x2, y2)
                        )  # 조건을 만족하면 저장

                # 결과 출력
                print("코트 내에 있는 사람 바운딩 박스:", filtered_person_boxes)

                # 코트 영역 내의 오른발과 왼발 좌표를 저장할 리스트
                right_foot_in_court = []
                left_foot_in_court = []

                # 코트 영역 내의 오른발 좌표만 필터링
                for pixel in right_foot_pixel:
                    if is_within_court_area(pixel[0], pixel[1], court_points):
                        right_foot_in_court.append(pixel)

                # 코트 영역 내의 왼발 좌표만 필터링
                for pixel in left_foot_pixel:
                    if is_within_court_area(pixel[0], pixel[1], court_points):
                        left_foot_in_court.append(pixel)

                # 결과 출력
                print("오른발 (코트 영역 내 픽셀 좌표):", right_foot_in_court)
                print("왼발 (코트 영역 내 픽셀 좌표):", left_foot_in_court)

                # 모든 오른발 픽셀에 대해 검은 점 그리기
                for pixel in right_foot_in_court:
                    # 좌표를 정수로 변환
                    point_x = int(pixel[0])
                    point_y = int(pixel[1])

                    # 이미지에 점 그리기
                    cv2.circle(
                        annotated_frame,
                        (point_x, point_y),
                        radius=5,
                        color=(0, 0, 0),
                        thickness=-1,
                    )  # radius=5, 검은색 점

                # 모든 왼발 픽셀에 대해 검은 점 그리기
                for pixel in left_foot_in_court:
                    # 좌표를 정수로 변환
                    point_xx = int(pixel[0])
                    point_yy = int(pixel[1])

                    # 이미지에 점 그리기
                    cv2.circle(
                        annotated_frame,
                        (point_xx, point_yy),
                        radius=5,
                        color=(0, 0, 0),
                        thickness=-1,
                    )  # radius=5, 검은색 점

                # 결과 이미지 출력
                cv2.imshow("YOLOv8 Pose Detection", annotated_frame)

                key = cv2.waitKey(0) & 0xFF  # 키 입력 대기
                if key == ord("t"):  # 't' 키가 눌렸는지 확인
                    print("'t' 키가 눌렸습니다.")

                    # y축 값이 가장 작은 좌표 찾기
                    # 오른발과 왼발 리스트를 합쳐서 최소 y 값을 가진 발 좌표를 찾습니다.
                    all_feet_in_court = right_foot_in_court + left_foot_in_court
                    min_y_pixel = min(
                        all_feet_in_court, key=lambda pixel: pixel[1]
                    )  # y 값 기준으로 최소값 찾기

                    # 최소 y 값을 가진 사람의 오른발과 왼발 좌표 찾기
                    # right_foot_in_court와 left_foot_in_court에서 동일한 사람의 좌표인지 확인
                    min_right_foot = None
                    min_left_foot = None

                    for i, pixel in enumerate(right_foot_in_court):
                        if np.array_equal(pixel, min_y_pixel):
                            min_right_foot = pixel
                            min_left_foot = left_foot_in_court[
                                i
                            ]  # 동일 인덱스로 왼발 찾기

                    if min_right_foot is None:
                        for i, pixel in enumerate(left_foot_in_court):
                            if np.array_equal(pixel, min_y_pixel):
                                min_left_foot = pixel
                                min_right_foot = right_foot_in_court[
                                    i
                                ]  # 동일 인덱스로 오른발 찾기

                    # 최소 y 값을 가진 사람의 오른발과 왼발을 빨간색으로 색칠
                    if min_right_foot is not None and min_left_foot is not None:
                        # 오른발에 빨간색 점 그리기
                        cv2.circle(
                            annotated_frame,
                            (int(min_right_foot[0]), int(min_right_foot[1])),
                            radius=5,
                            color=(0, 0, 255),
                            thickness=-1,
                        )

                        # 왼발에 빨간색 점 그리기
                        cv2.circle(
                            annotated_frame,
                            (int(min_left_foot[0]), int(min_left_foot[1])),
                            radius=5,
                            color=(0, 0, 255),
                            thickness=-1,
                        )

                    # 2번째 사람 저장할거임

                    # 빨간색으로 색칠한 사람의 오른발과 왼발 좌표를 제외한 나머지 좌표 저장
                    right_foot_in_court2 = []
                    left_foot_in_court2 = []

                    for i, right_foot in enumerate(right_foot_in_court):
                        # 오른발 좌표가 최소 y 값 좌표와 같지 않은 경우
                        if not np.array_equal(right_foot, min_right_foot):
                            right_foot_in_court2.append(right_foot)  # 오른발 좌표 추가

                    for i, left_foot in enumerate(left_foot_in_court):
                        # 왼발 좌표가 최소 y 값 좌표와 같지 않은 경우
                        if not np.array_equal(left_foot, min_left_foot):
                            left_foot_in_court2.append(left_foot)  # 왼발 좌표 추가

                    # y축 값이 가장 작은 좌표 찾기
                    # 오른발과 왼발 리스트를 합쳐서 최소 y 값을 가진 발 좌표를 찾습니다.
                    all_feet_in_court = right_foot_in_court2 + left_foot_in_court2
                    min_y_pixel = min(
                        all_feet_in_court, key=lambda pixel: pixel[1]
                    )  # y 값 기준으로 최소값 찾기

                    # 최소 y 값을 가진 사람의 오른발과 왼발 좌표 찾기
                    # right_foot_in_court와 left_foot_in_court에서 동일한 사람의 좌표인지 확인
                    min_right_foot2 = None
                    min_left_foot2 = None

                    for i, pixel in enumerate(right_foot_in_court2):
                        if np.array_equal(pixel, min_y_pixel):
                            min_right_foot2 = pixel
                            min_left_foot2 = left_foot_in_court2[
                                i
                            ]  # 동일 인덱스로 왼발 찾기

                    if min_right_foot2 is None:
                        for i, pixel in enumerate(left_foot_in_court2):
                            if np.array_equal(pixel, min_y_pixel):
                                min_left_foot2 = pixel
                                min_right_foot2 = right_foot_in_court2[
                                    i
                                ]  # 동일 인덱스로 오른발 찾기

                    # 최소 y 값을 가진 사람의 오른발과 왼발을 빨간색으로 색칠
                    if min_right_foot2 is not None and min_left_foot2 is not None:
                        # 오른발에 빨간색 점 그리기
                        cv2.circle(
                            annotated_frame,
                            (int(min_right_foot2[0]), int(min_right_foot2[1])),
                            radius=5,
                            color=(0, 0, 255),
                            thickness=-1,
                        )

                        # 왼발에 빨간색 점 그리기
                        cv2.circle(
                            annotated_frame,
                            (int(min_left_foot2[0]), int(min_left_foot2[1])),
                            radius=5,
                            color=(0, 0, 255),
                            thickness=-1,
                        )

                    # 3번째 사람

                    # 두 쌍의 최소 발 좌표를 제외한 나머지 좌표 저장
                    right_foot_in_court3 = []
                    left_foot_in_court3 = []

                    for i, right_foot in enumerate(right_foot_in_court):
                        # 오른발 좌표가 min_right_foot와 min_right_foot2와 같지 않은 경우
                        if not np.array_equal(
                            right_foot, min_right_foot
                        ) and not np.array_equal(right_foot, min_right_foot2):
                            right_foot_in_court3.append(right_foot)  # 오른발 좌표 추가

                    for i, left_foot in enumerate(left_foot_in_court):
                        # 왼발 좌표가 min_left_foot와 min_left_foot2와 같지 않은 경우
                        if not np.array_equal(
                            left_foot, min_left_foot
                        ) and not np.array_equal(left_foot, min_left_foot2):
                            left_foot_in_court3.append(left_foot)  # 왼발 좌표 추가

                    # y축 값이 가장 작은 좌표 찾기
                    # 오른발과 왼발 리스트를 합쳐서 최소 y 값을 가진 발 좌표를 찾습니다.
                    all_feet_in_court = right_foot_in_court3 + left_foot_in_court3
                    min_y_pixel = min(
                        all_feet_in_court, key=lambda pixel: pixel[1]
                    )  # y 값 기준으로 최소값 찾기

                    # 최소 y 값을 가진 사람의 오른발과 왼발 좌표 찾기
                    # right_foot_in_court와 left_foot_in_court에서 동일한 사람의 좌표인지 확인
                    min_right_foot3 = None
                    min_left_foot3 = None

                    for i, pixel in enumerate(right_foot_in_court3):
                        if np.array_equal(pixel, min_y_pixel):
                            min_right_foot3 = pixel
                            min_left_foot3 = left_foot_in_court3[
                                i
                            ]  # 동일 인덱스로 왼발 찾기

                    if min_right_foot3 is None:
                        for i, pixel in enumerate(left_foot_in_court3):
                            if np.array_equal(pixel, min_y_pixel):
                                min_left_foot3 = pixel
                                min_right_foot3 = right_foot_in_court3[
                                    i
                                ]  # 동일 인덱스로 오른발 찾기

                    # 최소 y 값을 가진 사람의 오른발과 왼발을 빨간색으로 색칠
                    if min_right_foot3 is not None and min_left_foot3 is not None:
                        # 오른발에 빨간색 점 그리기
                        cv2.circle(
                            annotated_frame,
                            (int(min_right_foot3[0]), int(min_right_foot3[1])),
                            radius=5,
                            color=(0, 0, 255),
                            thickness=-1,
                        )

                        # 왼발에 빨간색 점 그리기
                        cv2.circle(
                            annotated_frame,
                            (int(min_left_foot3[0]), int(min_left_foot3[1])),
                            radius=5,
                            color=(0, 0, 255),
                            thickness=-1,
                        )

                    right_foot_in_court4 = []
                    left_foot_in_court4 = []

                    for i, right_foot in enumerate(right_foot_in_court):
                        # 오른발 좌표가 min_right_foot, min_right_foot2, min_right_foot3와 같지 않은 경우
                        if (
                            not np.array_equal(right_foot, min_right_foot)
                            and not np.array_equal(right_foot, min_right_foot2)
                            and not np.array_equal(right_foot, min_right_foot3)
                        ):
                            right_foot_in_court4.append(right_foot)  # 오른발 좌표 추가

                    for i, left_foot in enumerate(left_foot_in_court):
                        # 왼발 좌표가 min_left_foot, min_left_foot2, min_left_foot3와 같지 않은 경우
                        if (
                            not np.array_equal(left_foot, min_left_foot)
                            and not np.array_equal(left_foot, min_left_foot2)
                            and not np.array_equal(left_foot, min_left_foot3)
                        ):
                            left_foot_in_court4.append(left_foot)  # 왼발 좌표 추가

                    # 추가 좌표 리스트
                    extra_feet = []

                    # 오른발 좌표 추가
                    extra_feet.extend(right_foot_in_court4)

                    # 왼발 좌표 추가
                    extra_feet.extend(left_foot_in_court4)

                    # 업데이트된 프레임 출력
                    cv2.imshow("YOLOv8 Pose Detection", annotated_frame)

                key = cv2.waitKey(0) & 0xFF  # 키 입력 대기
                if key == ord("y"):  # 'y' 키가 눌렸는지 확인
                    print("'y' 키가 눌렸습니다.")
                    print(
                        min_left_foot,
                        min_right_foot,
                        min_left_foot2,
                        min_right_foot2,
                        min_left_foot3,
                        min_right_foot3,
                    )
                    # 발 좌표 리스트
                    all_feet = [
                        min_left_foot,
                        min_right_foot,
                        min_left_foot2,
                        min_right_foot2,
                        min_left_foot3,
                        min_right_foot3,
                    ]

                    # 여기서 부터 왼선수 찾기

                    # x좌표가 가장 작은 좌표 찾기
                    min_x_pixel = min(
                        all_feet, key=lambda pixel: pixel[0]
                    )  # x 값 기준으로 최소값 찾기

                    # 결과 출력
                    print("x좌표가 가장 작은 좌표:", min_x_pixel)

                    index_of_min_x_pixel = 0  # 초기 최소값 인덱스

                    # min_x_pixel이 all_feet의 어떤 값인지 확인하기
                    for i in range(1, len(all_feet)):
                        if (
                            all_feet[i][0] < all_feet[index_of_min_x_pixel][0]
                        ):  # x좌표가 더 작으면 인덱스 갱신
                            index_of_min_x_pixel = i

                    print("min_x_pixel의 인덱스:", index_of_min_x_pixel)

                    if index_of_min_x_pixel in [0, 1]:  # 0 또는 1인 경우
                        Front_Left_foot = [all_feet[0], all_feet[1]]
                        Rest_foot = [all_feet[2], all_feet[3], all_feet[4], all_feet[5]]

                    elif index_of_min_x_pixel in [2, 3]:  # 2 또는 3인 경우
                        Front_Left_foot = [all_feet[2], all_feet[3]]
                        Rest_foot = [all_feet[0], all_feet[1], all_feet[4], all_feet[5]]

                    elif index_of_min_x_pixel in [4, 5]:  # 4 또는 5인 경우
                        Front_Left_foot = [all_feet[4], all_feet[5]]
                        Rest_foot = [all_feet[0], all_feet[1], all_feet[2], all_feet[3]]

                    else:
                        print("오류남")

                    print(Front_Left_foot)

                    text_position1 = (
                        int(Front_Left_foot[0][0]),
                        int(Front_Left_foot[0][1]),
                    )  # 첫 번째 발 좌표의 (x, y)

                    # Define the text to be added
                    text1 = "1"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    color = (255, 255, 255)  # White color
                    thickness = 2

                    # Add text to the image
                    cv2.putText(
                        annotated_frame,
                        text1,
                        text_position1,
                        font,
                        font_scale,
                        color,
                        thickness,
                    )
                    # Add text to the image
                    cv2.putText(
                        last_frame,
                        text1,
                        text_position1,
                        font,
                        font_scale,
                        color,
                        thickness,
                    )

                    # 여기서 부터 오른선수 찾기

                    # x좌표가 가장 큰 좌표 찾기
                    max_x_pixel = max(
                        Rest_foot, key=lambda pixel: pixel[0]
                    )  # x 값 기준으로 최소값 찾기

                    # 결과 출력
                    print("x좌표가 가장 큰 좌표:", max_x_pixel)

                    index_of_max_x_pixel = 0  # 초기 최소값 인덱스

                    # min_x_pixel이 all_feet의 어떤 값인지 확인하기
                    for i in range(1, len(Rest_foot)):
                        if (
                            Rest_foot[i][0] > Rest_foot[index_of_max_x_pixel][0]
                        ):  # x좌표가 더 작으면 인덱스 갱신
                            index_of_max_x_pixel = i

                    print("max_x_pixel의 인덱스:", index_of_max_x_pixel)

                    if index_of_max_x_pixel in [0, 1]:  # 0 또는 1인 경우
                        Front_Right_foot = [Rest_foot[0], Rest_foot[1]]
                        Front_Middle_foot = [
                            Rest_foot[2],
                            Rest_foot[3],
                        ]

                    elif index_of_max_x_pixel in [2, 3]:  # 2 또는 3인 경우
                        Front_Right_foot = [Rest_foot[2], Rest_foot[3]]
                        Front_Middle_foot = [
                            Rest_foot[0],
                            Rest_foot[1],
                        ]

                    else:
                        print("오류남")

                    print(Front_Right_foot)
                    print(Front_Middle_foot)
                    text_position2 = (
                        int(Front_Middle_foot[0][0]),
                        int(Front_Middle_foot[0][1]),
                    )  # 첫 번째 발 좌표의 (x, y)
                    text_position3 = (
                        int(Front_Right_foot[0][0]),
                        int(Front_Right_foot[0][1]),
                    )  # 첫 번째 발 좌표의 (x, y)
                    text2 = "2"
                    text3 = "3"

                    # Add text to the image
                    cv2.putText(
                        annotated_frame,
                        text2,
                        text_position2,
                        font,
                        font_scale,
                        color,
                        thickness,
                    )
                    # Add text to the image
                    cv2.putText(
                        last_frame,
                        text2,
                        text_position2,
                        font,
                        font_scale,
                        color,
                        thickness,
                    )
                    # Add text to the image
                    cv2.putText(
                        annotated_frame,
                        text3,
                        text_position3,
                        font,
                        font_scale,
                        color,
                        thickness,
                    )
                    cv2.putText(
                        last_frame,
                        text3,
                        text_position3,
                        font,
                        font_scale,
                        color,
                        thickness,
                    )

                    # 여기서 부터 후위선수 나누기

                    # 여기서 부터 후위 왼선수 찾기

                    # x좌표가 가장 작은 좌표 찾기
                    bmin_x_pixel = min(
                        extra_feet, key=lambda pixel: pixel[0]
                    )  # x 값 기준으로 최소값 찾기

                    bindex_of_min_x_pixel = 0  # 초기 최소값 인덱스

                    # bmin_x_pixel이 extra_feet의 어떤 값인지 확인하기
                    for i in range(1, len(extra_feet)):
                        if (
                            extra_feet[i][0] < extra_feet[bindex_of_min_x_pixel][0]
                        ):  # x좌표가 더 작으면 인덱스 갱신
                            bindex_of_min_x_pixel = i

                    print("후위 왼쪽의 인덱스:", bindex_of_min_x_pixel)
                    print("후위 왼쪽의 값:", extra_feet[bindex_of_min_x_pixel][0])

                    if bindex_of_min_x_pixel in [0, 3]:
                        Back_Left_foot = [extra_feet[0], extra_feet[3]]
                        bRest_foot = [
                            extra_feet[1],
                            extra_feet[2],
                            extra_feet[4],
                            extra_feet[5],
                        ]
                        print("1번")

                    elif bindex_of_min_x_pixel in [1, 4]:
                        Back_Left_foot = [extra_feet[1], extra_feet[4]]
                        bRest_foot = [
                            extra_feet[0],
                            extra_feet[2],
                            extra_feet[3],
                            extra_feet[5],
                        ]
                        print("2번")

                    elif bindex_of_min_x_pixel in [2, 5]:
                        Back_Left_foot = [extra_feet[2], extra_feet[5]]
                        bRest_foot = [
                            extra_feet[0],
                            extra_feet[1],
                            extra_feet[3],
                            extra_feet[4],
                        ]
                        print("3번")

                    else:
                        print("오류남")

                    print("후위 왼쪽")
                    print(Back_Left_foot)

                    text_position4 = (
                        int(Back_Left_foot[1][0]),
                        int(Back_Left_foot[1][1]),
                    )  # 네 번째 발 좌표의 (x, y)

                    text4 = "4"

                    # Add text to the image
                    cv2.putText(
                        annotated_frame,
                        text4,
                        text_position4,
                        font,
                        font_scale,
                        color,
                        thickness,
                    )
                    cv2.putText(
                        last_frame,
                        text4,
                        text_position4,
                        font,
                        font_scale,
                        color,
                        thickness,
                    )

                    # 여기서 부터 후위 오른과 중앙 선수 찾기

                    # x좌표가 가장 큰 좌표 찾기
                    bmax_x_pixel = max(
                        bRest_foot, key=lambda pixel: pixel[0]
                    )  # x 값 기준으로 최소값 찾기

                    bbindex_of_max_x_pixel = 0  # 초기 최소값 인덱스

                    # min_x_pixel이 all_feet의 어떤 값인지 확인하기
                    for i in range(1, len(bRest_foot)):
                        if (
                            bRest_foot[i][0] > bRest_foot[bbindex_of_max_x_pixel][0]
                        ):  # x좌표가 더 작으면 인덱스 갱신
                            bbindex_of_max_x_pixel = i

                    if bbindex_of_max_x_pixel in [0, 2]:
                        Back_Right_foot = [bRest_foot[0], bRest_foot[2]]
                        Back_Middle_foot = [
                            bRest_foot[1],
                            bRest_foot[3],
                        ]

                    elif bbindex_of_max_x_pixel in [1, 3]:
                        Back_Right_foot = [bRest_foot[1], bRest_foot[3]]
                        Back_Middle_foot = [
                            bRest_foot[0],
                            bRest_foot[2],
                        ]

                    else:
                        print("오류남")

                    print("후위의 오른 미들")
                    print(Back_Right_foot)

                    print(Back_Middle_foot)

                    text_position5 = (
                        int(Back_Middle_foot[1][0]),
                        int(Back_Middle_foot[1][1]),
                    )  # 첫 번째 발 좌표의 (x, y)
                    text_position6 = (
                        int(Back_Right_foot[1][0]),
                        int(Back_Right_foot[1][1]),
                    )  # 첫 번째 발 좌표의 (x, y)
                    text5 = "5"
                    text6 = "6"

                    # Add text to the image
                    cv2.putText(
                        annotated_frame,
                        text5,
                        text_position5,
                        font,
                        font_scale,
                        color,
                        thickness,
                    )
                    cv2.putText(
                        last_frame,
                        text5,
                        text_position5,
                        font,
                        font_scale,
                        color,
                        thickness,
                    )
                    # Add text to the image
                    cv2.putText(
                        annotated_frame,
                        text6,
                        text_position6,
                        font,
                        font_scale,
                        color,
                        thickness,
                    )
                    cv2.putText(
                        last_frame,
                        text6,
                        text_position6,
                        font,
                        font_scale,
                        color,
                        thickness,
                    )

                    # 업데이트된 프레임 출력
                    cv2.imshow("YOLOv8 Pose Detection", annotated_frame)

                key = cv2.waitKey(0) & 0xFF  # 키 입력 대기
                if key == ord("u"):  # 'u' 키가 눌렸는지 확인
                    print("'u' 키가 눌렸습니다.")
                    ## 2는 shirt 3은 숫자인식
                    model_2 = YOLO("C:/Users/PC/Desktop/배구/CNN/custom_shirt_2.pt")
                    model_3 = YOLO(
                        "C:/Users/PC/Desktop/배구/CNN/custom_backnum_4with_test.pt"
                    )
                    results_2 = model_2(frame_custom, conf=0.1)
                    annotated_frame = results_2[0].plot()

                    detected_objects = []  # 감지된 객체 정보를 저장할 리스트

                    for i, box in enumerate(results_2[0].boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cropped_box = frame[y1:y2, x1:x2]

                        # 감지 임계값 설정 (conf=0.1로 설정하여 더 많은 객체 감지)
                        results_3 = model_3(cropped_box)

                        # 감지된 객체의 결과 출력
                        if results_3 and results_3[0].boxes.cls.numel() > 0:
                            for class_id in results_3[
                                0
                            ].boxes.cls:  # 여러 클래스 ID가 있는 경우 반복 처리
                                class_name = model_3.names[
                                    int(class_id)
                                ]  # 클래스 이름을 가져옴
                                print(
                                    f"클래스 ID: {int(class_id)}, 클래스 이름: {class_name}"
                                )
                        else:
                            print("감지된 객체가 없습니다.")

                        # 중심 좌표 계산
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        # 감지된 클래스 이름과 중심 좌표를 딕셔너리에 저장하여 리스트에 추가
                        detected_objects.append(
                            {"class_name": class_name, "center": (center_x, center_y)}
                        )

                    # 결과 확인
                    for obj in detected_objects:
                        print(f"Detected Class: {obj['class_name']}")
                        print(f"Center Coordinates: {obj['center']}")

                    cv2.imshow("YOLOv8 Pose Detection", annotated_frame)

                class_names = []  # 반복문 외부에서 class_names 배열을 초기화

                key = cv2.waitKey(0) & 0xFF  # 키 입력 대기
                if key == ord("i"):  # 'i' 키가 눌리면 루프 종료
                    print("'i' 키가 눌렸습니다.")

                    positions = [
                        text_position1,
                        text_position2,
                        text_position3,
                        text_position4,
                        text_position5,
                        text_position6,
                    ]  # 다 왼발이다.

                    # 각 text_position에 대해 처리
                    for pos_index, text_position in enumerate(positions, start=1):
                        a = None  # a 값 초기화
                        x, y = text_position  # text_position 좌표

                        # filtered_person_boxes 안에서 해당 text_position이 있는지 확인
                        for i, (x1, y1, x2, y2) in enumerate(filtered_person_boxes):
                            if x1 <= x <= x2 and (y1 + y2) / 2 <= y <= y2:
                                a = i  # a 값을 저장

                        # 만약 a 값이 None이 아니면 해당 인덱스만 처리
                        if a is not None:
                            # 해당 인덱스의 filtered_person_boxes만 수행
                            x1, y1, x2, y2 = filtered_person_boxes[a]
                            best_obj = (
                                None  # 가장 높은 confidence 값을 가진 객체를 저장
                            )

                            for j, obj in enumerate(detected_objects):
                                center_x, center_y = obj["center"]  # 중심 좌표 가져오기
                                if (
                                    x1 <= center_x <= x2 and y1 <= center_y <= y2
                                ) and obj[
                                    "class_name"
                                ] != "person":  # 'person' 클래스는 제외
                                    # 처음 또는 더 높은 confidence 값을 가진 객체를 선택
                                    if (
                                        best_obj is None
                                        or obj["confidence"] > best_obj["confidence"]
                                    ):
                                        best_obj = (
                                            obj  # 가장 높은 confidence를 가진 객체 저장
                                        )

                            # 가장 높은 confidence 값을 가진 객체가 있다면 출력
                            if best_obj is not None:
                                class_names.append(
                                    best_obj["class_name"]
                                )  # class_name 추가
                                print(
                                    f"{pos_index}번 포지션의 등 번호: {best_obj['class_name']}"
                                )

                    # class_names 배열 출력 (혹은 이후 사용)
                    print("저장된 class_names:", class_names)

                    # 업데이트된 프레임 출력
                    cv2.imshow("YOLOv8 Pose Detection", last_frame)

                key = cv2.waitKey(0) & 0xFF  # 키 입력 대기
                if key == ord("o"):  # 'o' 키가 눌리면 루프 종료
                    print("'o' 키가 눌렸습니다.")

                    # 이미지 파일 경로
                    image_court_path = "C:/Users/PC/Desktop/배구/이미지/position.jpg"

                    # 이미지 읽기
                    img_court = mpimg.imread(image_court_path)

                    # 숫자 배열
                    numbers = class_names

                    # 텍스트가 위치할 x, y 좌표 배열
                    positions_court = [
                        (105, 355),  # 첫 번째 숫자 위치
                        (260, 355),  # 두 번째 숫자 위치
                        (420, 355),  # 세 번째 숫자 위치
                        (105, 460),  # 네 번째 숫자 위치
                        (260, 460),  # 다섯 번째 숫자 위치
                        (420, 460),  # 여섯 번째 숫자 위치
                    ]

                    # 이미지 보여주기
                    fig, ax = plt.subplots()
                    ax.imshow(img_court)
                    ax.axis("off")  # 축을 숨김

                    # 숫자 배열을 반복하여 텍스트 추가
                    for i, num in enumerate(numbers):
                        x, y = positions_court[i]
                        ax.text(
                            x,
                            y,
                            str(num),
                            color="black",
                            fontsize=40,
                            fontweight="bold",
                            ha="center",
                            va="center",
                        )

                    # 이미지 출력
                    plt.show()

            while True:
                key = cv2.waitKey(0) & 0xFF  # 키 입력 대기
                if key == ord("p"):  # 'p' 키가 눌리면 루프 종료
                    print("'p' 키가 눌렸습니다.")
                    plt.close()
                    break  # 'p' 키가 눌리면 루프 종료

        elif key == 27:  # ESC 키로 종료
            break

    # 캡처 객체 해제 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()
