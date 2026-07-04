# Video를 Scene + G1 Motion NPZ로 변환하는 절차

이 문서는 입력 비디오를 CRISP real2sim 결과로 변환한 뒤, PyRoki로 Unitree G1 29-DoF motion에 retarget하고, 학습용 `scene + g1 motion` `.npz` 파일과 검증용 MP4를 만드는 절차를 정리한다.

## 최종 산출물

예시 sequence:

```text
climbing_down_stairs_001
```

최종 dataset:

```text
results/output/scene_g1_motion_dataset/climbing_down_stairs_001/gv/climbing_down_stairs_001_scene_g1_motion.npz
```

검증 영상:

```text
results/output/scene_g1_motion_dataset/climbing_down_stairs_001/gv/climbing_down_stairs_001_scene_g1_motion.mp4
```

## 1. CRISP Real2Sim 실행 결과

비디오는 먼저 CRISP pipeline을 통해 `post_scene` 결과로 변환되어야 한다. 이 단계의 핵심 입력은 비디오이고, 핵심 출력은 정렬된 scene mesh와 SMPL human motion이다.

필요한 post-scene 파일:

```text
results/output/post_scene/<seq_name>/<hmr_type>/scene_mesh_sqs/scene_mesh_sqs.obj
results/output/post_scene/<seq_name>/<hmr_type>/hmr/human_motion.npz
results/output/post_scene/<seq_name>/<hmr_type>/hmr/<seq_name>.npz
```

좌표계는 `world_z_up_scene_grounded_gravity_down_negative_z`로 취급한다. 즉 `+Z`가 위쪽이고 중력은 `[0, 0, -9.81]`이다. retarget 전에 scene mesh의 최저 z가 `0` 근처인지, `world_rotation`이 정상 rotation matrix인지 확인한다.

## 2. G1 Retarget 실행

현재 안정적인 retarget 방법은 PyRoki의 `simple` mode다. `fancy` mode는 CRISP SQS mesh 전체를 heightmap 접촉/충돌 비용에 사용하므로, 계단/벽/배경 평면을 잘못 ground로 잡아 root가 위아래로 튈 수 있다. 앞으로 retarget 요청이 있으면 기본적으로 `simple` mode를 사용한다.

실행 예시:

```bash
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n pyroki python scripts/retarget_crisp_scene_to_g1_pyroki.py \
  --seq-name climbing_down_stairs_001 \
  --hmr-type gv \
  --mode simple \
  --force
```

생성 파일:

```text
results/output/g1_pyroki/<seq_name>/<hmr_type>/<seq_name>_scene_g1_pyroki_simple.npz
```

이 파일에는 PyRoki retarget 결과와 일부 중간 metadata가 포함된다.

## 3. 학습용 Dataset Export

retarget 결과를 portable dataset으로 정리한다. 학습용 `.npz`에는 scene triangle mesh와 G1 motion, fps/dt/좌표계 metadata만 남긴다.

```bash
conda run -n pyroki python scripts/export_scene_g1_motion_dataset.py \
  --seq-name climbing_down_stairs_001 \
  --hmr-type gv \
  --retarget-mode simple \
  --force
```

주요 저장 배열:

```text
scene_mesh_vertices: (V, 3)
scene_mesh_faces: (F, 3)
g1_root_pos: (T, 3)
g1_root_quat_wxyz: (T, 4)
g1_dof_pos: (T, 29)
g1_dof_vel: (T, 29)
g1_qpos: (T, 36)
fps, dt, frame_count
```

세부 schema는 `docs/scene_g1_motion_dataset_ko.md`를 따른다.

## 4. MP4 시각화 검증

dataset이 만들어지면 scene mesh 위에 G1 skeleton을 렌더링해 motion을 확인한다.

```bash
JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n pyroki python scripts/visualize_scene_g1_motion_dataset.py \
  --dataset results/output/scene_g1_motion_dataset/climbing_down_stairs_001/gv/climbing_down_stairs_001_scene_g1_motion.npz \
  --output results/output/scene_g1_motion_dataset/climbing_down_stairs_001/gv/climbing_down_stairs_001_scene_g1_motion.mp4 \
  --fps 30 \
  --stride 1
```

검증 기준은 G1 root 높이가 SMPL pelvis 높이와 비슷하게 움직이고, 프레임 간 z 변화가 비정상적으로 크지 않은 것이다. 예시 sequence의 안정 결과는 G1 root z 평균이 약 `1.60m`, 프레임 간 z 변화가 약 `-0.024 ~ +0.008m`였다.

## 주의사항

`*_stable.npz`와 `*_stable.mp4`는 과거에 동일 결과를 복사해 둔 중복 파일이었으므로 삭제했다. 앞으로는 suffix 없는 최종 파일만 사용한다.

`fancy` retarget은 terrain-only heightmap이 별도로 준비되기 전까지 사용하지 않는다. scene-aware 접촉 비용이 필요하면 CRISP SQS mesh 전체가 아니라 simulator에서 사용할 지형/계단 collision mesh를 별도로 추출한 뒤 적용해야 한다.
