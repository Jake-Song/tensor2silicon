# 주방 비유로 배우는 GPU 계산과 데이터 이동

초보자용 대본과 26개 새 장면으로 만든 별도 영상입니다. 요리사·재료 창고·운반을
계산 장치·메모리·데이터 이동에 대응시키고, 용어의 뜻을 설명한 뒤 숫자 예제를 풉니다.
기존 MP4나 장면 클립을 입력으로 사용하지 않습니다. 모든 도식과 애니메이션은
`visuals.py`가 새로 그립니다. 밝은 배경과 청록색 데이터·주황색 계산·초록색 결과를 사용합니다.

- 완성 영상: `../week3-beginner.ko.mp4` — 10분, 1920×1080, 30fps, H.264/AAC.
- 미리보기: `../week3-beginner-preview.ko.mp4` — 처음 60초.
- 읽을 대본: `script.ko.md` — 실제 장면 시작 시각 포함.
- 대본 원본: `lesson.json` — 26개 장면의 내레이션·제목·핵심 문구.
- 음성: 기존 여성 목소리 `ko-KR-SunHiNeural` (SunHi, edge-tts).
- 자막: 새 음성의 단어 시각에서 생성한 `week3.ko.srt`, 영상에도 표시.

7개 챕터: 주방으로 이해하기, 계산과 단위, 창고와 운반, 두 가지 시간,
숫자로 함께 풀기, 무엇을 바꾸면 빨라질까, 확인하고 정리하기.

## 재생성

프로젝트 루트에서 실행합니다. 음성 합성에는 인터넷과 edge-tts가 필요합니다.
렌더링에는 Pillow, FFmpeg, 은돋움·DejaVu 폰트를 사용합니다.

```bash
uv run --with edge-tts python week3/video/beginner/build.py audio
.venv/bin/python week3/video/beginner/build.py preview
.venv/bin/python week3/video/beginner/build.py render
.venv/bin/python week3/video/beginner/build.py assemble
.venv/bin/python week3/video/beginner/build.py check
```

`render --scene 0`으로 한 장면만 만들 수 있습니다. 완료된 새 음성과 새 장면은
대본·설정·코드 해시로 캐시합니다. `build.py audio`는 `../intuition/produce.py`의
음성 합성·시간 배분 유틸리티만 사용하며, 기존 음성이나 영상은 읽지 않습니다.
`sync`는 합성이 끝난 음성을 장면에 맞춰 결합하고 챕터 파일을 만듭니다.

전체 음성은 음높이를 유지하며 약 1.205배로 조정하고, 장면 사이에 쉼을 둡니다.
복습 정답과 240배 강조는 새 발화 시점에 표시합니다. 대본을 바꾸면 음성·자막·
타이밍과 장면을 다시 생성해야 합니다. `check`는 대본과 음성 ID, 캐시 해시,
자막 순서, 10분 PCM 길이, 18,000프레임, 챕터 시각과 전체 디코딩을 확인합니다.

수치 예제는 실제 제품 사양이 아닌 가상 조건입니다. 입력이 이미 HBM에 있고,
공간이 충분하며 입력을 한 번씩 읽고 새 결과를 한 번 쓴다고 가정합니다.
계산 5 μs와 이동 1.2 ms를 비교하며, 충분한 중첩과 성능 활용을 전제로
더 긴 시간이 기준이 됩니다. 주방 그림은 역할을 설명하는 비유입니다.
