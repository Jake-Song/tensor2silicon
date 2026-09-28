# 기존 2D 영상 제작 기록

## 수학적 직관 중심 개선본

`week3-intuition.ko.mp4`는 기존 10분 한국어 내레이션을 유지하면서 모든 장면을
새 2D 애니메이션으로 렌더링한 버전입니다. 1920×1080, 30fps, H.264/AAC이며
한국어 자막과 챕터가 포함됩니다. `week3-intuition-preview.ko.mp4`는 벡터 덧셈,
240배 시간 비교, 병목 변화의 세 장면을 담은 45초 미리보기입니다.

검은 배경, 선과 수식 중심 도식, 청록색 데이터 이동·노란색 연산·초록색 결과를
사용합니다. 원소별 덧셈, 분수의 기호에서 수치로의 전환, 읽기·계산·쓰기 중첩,
대역폭 변화에 따른 막대 길이 변화가 순차적으로 진행됩니다. 시간 막대는 실제
비율을 유지하며, 용량 비교도 같은 너비와 두 배 높이로 표현합니다. 복습의 답과
240배 강조는 기존 음성 경계 시각을 이용합니다. 특정 채널의 로고·캐릭터는 쓰지 않습니다.

개선본의 편집 원본은 `intuition.py`, `timeline.json`과 연결된 음성입니다.
2D 버전은 Python으로 재생성합니다.

```bash
# Pillow 12.3, FFmpeg(libass 포함), 은돋움·DejaVu 폰트 필요
.venv/bin/python week3/video/intuition.py preview
.venv/bin/python week3/video/intuition.py render
.venv/bin/python week3/video/intuition.py assemble
.venv/bin/python week3/video/intuition.py check
```

`render --scene 21`로 한 장면을 렌더할 수 있습니다. 프레임은 FFmpeg에 바로
전달하며 디스크에 18,000장의 이미지를 저장하지 않습니다. 코드·타임라인·음성
경계·폰트의 해시가 바뀌면 클립을 자동 재생성합니다. `intuition/preview/`에
30개 대표 화면과 주요 장면의 시간별 화면을 저장합니다. `check`는 최종 파일을
전체 디코딩하고 길이·프레임 수·해상도·음성·챕터·캐시 최신 여부를 확인합니다.
