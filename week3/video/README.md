# 3주차 설명 영상

## Transformer 연산 지도

[Transformer의 파라미터와 FLOPs](./transformer/transformer-ops.ko.mp4)는
저장소 LLM의 한 층을 펼쳐 9개 matmul과 정규화·RoPE·softmax·원소별 연산을
설명하는 10분 영상입니다. 어두운 2D 도식, 한국어 SunHi 음성·자막을 제공합니다.
전체 73개 matmul과 약 5.42억 파라미터를 합산하고 prefill·decode도 비교합니다.

- [첫 60초 미리보기](./transformer/transformer-ops-preview.ko.mp4)
- [대본](./transformer/script.ko.md) · [제작·계산 기준](./transformer/README.md)

## 초보자용 새 영상

[주방 비유로 배우는 GPU 계산과 데이터 이동](./week3-beginner.ko.mp4)은
새 대본과 26개의 새 장면으로 제작한 10분 영상입니다. 요리사·재료 창고·운반의
비유로 계산, 메모리 용량과 대역폭, 병목을 설명합니다. 기존 여성 목소리
**SunHi (`ko-KR-SunHiNeural`)**를 사용합니다.

- [처음 60초 미리보기](./week3-beginner-preview.ko.mp4)
- [새 대본](./beginner/script.ko.md)
- [장면·대본 원본](./beginner/lesson.json)
- [제작·재생성 안내](./beginner/README.md)

밝은 배경의 새 도식과 애니메이션으로 구성했습니다. 기존 영상이나 클립을
입력으로 사용하지 않습니다. 1920×1080, 30fps, 한국어 음성·자막, 7개 챕터를 제공합니다.

## 기존 2D 영상

[기존 10분 영상](./week3-intuition.ko.mp4)과
[기존 45초 미리보기](./week3-intuition-preview.ko.mp4)는 별도로 유지합니다.
기존 대본은 `script.ko.md`, 타임라인은 `timeline.json`, 렌더링 코드는 `intuition.py`입니다.
새 영상의 대본과 제작 자료는 `beginner/`에 있습니다.

## 정리한 파일

사용자 요청으로 모든 ZIP 파일과 ElevenLabs 전용 코드·음성·캐시·백업·로그·검증 결과를 삭제했습니다.
Blender 영상·원본·백업·전용 코드·렌더 캐시·미리보기 이미지·로그·검증 결과도 삭제했습니다.
2D 영상 제작에 사용하는 공통 코드·대본·SunHi 음성은 유지합니다.
이전 제작 절차는 [기록용 문서](./legacy-production.md)에 남겼습니다.

MP4·음성·렌더 이미지·ZIP은 로컬 산출물이며 Git에서 제외합니다.
