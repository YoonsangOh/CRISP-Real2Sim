# VideoMimic 계단 비디오를 Scene + G1 Motion NPZ로 변환하는 배치 절차

이 문서는 `climbing_down_stairs_001`, `002`, `003`에서 검증한 절차를 기준으로, 비디오를 CRISP real2sim 결과와 PyRoki G1 retarget 결과가 포함된 학습용 `.npz`로 만드는 방법을 정리한다.

## 검증된 산출물

최종 파일은 다음 위치에 저장된다.

```text
results/output/scene_g1_motion_dataset/<seq>/gv/<seq>_scene_g1_motion.npz
```

현재 검증된 예시는 다음 3개다.

```text
climbing_down_stairs_001: CRISP SQS mesh + PyRoki simple retarget
climbing_down_stairs_002: CRISP SQS mesh + PyRoki simple retarget
climbing_down_stairs_003: CRISP SQS mesh + PyRoki simple retarget
```

`003`의 이전 실패 원인은 OOM이 아니라 `results/init/flows/climbing_down_stairs_003` 누락이었다. `0_ufm.sh`가 `CRISP_GPU_IDS=5`처럼 물리 GPU 번호 하나를 받을 때 작업 분배 인덱스를 잘못 계산해 UFM flow/covisibility 파일을 만들지 않았고, SQS 단계가 frame 간 segment correspondence 없이 실행되어 primitive 0개를 출력했다. 수정 후 `003`은 82쌍의 flow/covisibility, 176개 correspondence, 26개 SQS primitive를 생성한다.

## 단일 비디오 실행

반드시 `gpustat`로 빈 GPU를 확인한 뒤 명시한다. 스크립트도 `memory.used < 1000MiB`가 아니면 실행을 중단한다.

```bash
GPU_ID=5 bash scripts/run_video_to_scene_g1_motion.sh \
  videomimic_videos/climbing_down_stairs_003.mp4
```

처리 단계는 다음 순서다.

```text
video -> CRISP real2sim -> post_scene scene/human motion
post_scene mesh/human motion 검증
PyRoki simple G1 retarget
scene + g1 motion npz export
```

`fancy` retarget은 사용하지 않는다. CRISP scene mesh 전체를 contact/collision heightmap으로 쓰면 계단/벽/배경 평면 때문에 root가 튈 수 있으므로, 안정적인 `simple` mode만 사용한다.

## 전체 배치 실행

계단 비디오 전체를 순차 처리하려면 다음처럼 실행한다.

```bash
GPU_ID=5 VIDEO_PATTERN='*stairs*.mp4' \
LOG_DIR=logs/batch_stairs_scene_g1_motion \
SUMMARY_FILE=logs/batch_stairs_scene_g1_motion/summary.tsv \
bash scripts/batch_stairs_scene_g1_motion.sh
```

`climbing_down_stairs` 계열만 처리하려면:

```bash
GPU_ID=5 VIDEO_PATTERN='climbing_down_stairs*.mp4' \
LOG_DIR=logs/batch_climbing_down_scene_g1_motion \
SUMMARY_FILE=logs/batch_climbing_down_scene_g1_motion/summary.tsv \
bash scripts/batch_stairs_scene_g1_motion.sh
```

이미 최종 `.npz`가 있으면 `SKIP`으로 기록된다. 실패 시 `summary.tsv`에는 OOM, missing pointcloud, missing post_scene 등 핵심 원인을 로그에서 추출해 기록한다.

## 결과 검증 기준

각 `.npz`에는 다음 배열이 있어야 한다.

```text
scene_mesh_vertices, scene_mesh_faces
g1_root_pos, g1_root_quat_wxyz
g1_dof_pos, g1_dof_vel
fps, dt, frame_count
```

`003` 검증 결과는 `frames=161`, `fps=30`, `dof=29`, `scene_vertices=4212`, `scene_faces=8320`다. root z 변화는 약 `-0.023 ~ +0.008m/frame`로 001/002와 같은 안정 범위다.

## 주의사항

Fallback mesh는 batch 기본 경로에서 사용하지 않는다. `ALLOW_POINTCLOUD_MESH_FALLBACK=1`을 명시하지 않는 한 `post_scene` SQS mesh가 없으면 실패로 처리한다.
