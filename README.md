
1. 영상의 각도, 거리, 경기장 코트 규격에 따라 코트의 위치가 달라지므로, 시스템 시작 전에 코트의 끝점을 클릭하여 저장하고 디스플레이합니다.
![image](https://github.com/user-attachments/assets/69e17c0d-8e1f-442c-8d2c-aa343d8a2751)
![image](https://github.com/user-attachments/assets/1f8cef7a-cbf8-4a66-9a8e-2e25bb1985da)
⦁ 코트의 디스플레이는 영상(경기) 진행 중에도 유지되며, 서브 시 코트 안의 선수 발 위치를 판별하는 데 사용됩니다.

2. AI 기술을 활용하여 YOLO11 모델을 통해 사람을 인식하고, 바운딩 박스와 Pose를 검출합니다.
![image](https://github.com/user-attachments/assets/37ae584c-5153-4918-bb19-1f23c87fa8f5)

3. 이후 인식된 코트 안의 선수를 분리합니다.
(시각적 효과를 위해 발 위치에 검은색으로 표시합니다.)
![image](https://github.com/user-attachments/assets/78f497e7-7163-4654-ad91-d2b0690b5a37)

4. 이후 6명의 선수들 중에서 한 발이라도 앞에 있는 선수가 전위로 인식되며, 거리 계산을 통해 앞에 있는 발을 전위로 표시합니다. 
(시각적 효과를 위해 전위는 빨간색, 후위는 검은색으로 표시됩니다.)
![image](https://github.com/user-attachments/assets/226d522e-b847-40bf-b702-e9d6dc74746a)

5. 이후 전위 선수와 후위 선수 간의 좌측, 중간, 우측 위치를 구분하기 위해, 한 발이라도 왼쪽 코트에 가까운 선수와 한 발이라도 오른쪽 코트에 가까운 선수를 선별하여 선수들에게 번호를 매깁니다.
(시각적 효과를 위해 숫자로 표시하였습니다.)
![image](https://github.com/user-attachments/assets/78e1959b-c467-46dc-94d3-1051110c0bfb)

6. 이후 등번호의 위치를 라벨링한 데이터를 기반으로 약 1700장의 이미지를 학습시켜 등번호 위치를 탐지하는 AI 모델을 개발합니다. 이를 활용하여 등번호가 존재할 위치를 정확히 찾습니다.

⦁ 모델의 학습/검증/테스트 라벨링이미지의 예시
![image](https://github.com/user-attachments/assets/cc8e256a-2c51-4d80-8737-3fcffc98e668)
⦁ 살제 모델을 적용하여 검출된 이미지
![image](https://github.com/user-attachments/assets/60b4c327-bc31-4176-a096-03eaf988e96d)

7. 이후 추출된 위치의 이미지를 잘라내어 저장한 뒤, 약 7200장의 데이터를 활용해 등번호 0~99번을 학습하는 AI 모델을 개발 하였고, 이 모델을 통해 잘라낸 이미지를 숫자로 변환합니다.

⦁ 모델의 학습/검증/테스트 라벨링이미지의 예시
![image](https://github.com/user-attachments/assets/d0ea6f7e-1d70-433f-8795-8e6d606ff0b9)
⦁ 살제 모델을 적용하여 추출된 정보
![image](https://github.com/user-attachments/assets/8bf6873f-1928-4fd0-9682-cd39e13dfb83)

8. 마지막으로 선수의 위치, 발 위치, 추출한 등번호를 조합하여, 포지션 폴트 판정에 중요한 선수의 위치와 등번호를 간단히 확인할 수 있는 화면을 구성합니다.
![image](https://github.com/user-attachments/assets/953b9517-2c13-4209-9efc-c2378b76f088)




