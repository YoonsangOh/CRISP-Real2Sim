# Scene + G1 Motion Dataset 형식

이 문서는 CRISP real2sim 결과와 PyRoki G1 retarget 결과를 합쳐 만든 학습용 `.npz` 데이터셋 형식을 설명한다. 데이터셋은 terrain-aware motion tracking 프로젝트로 옮겨서 바로 로드할 수 있도록 scene mesh와 G1 29-DoF motion만 포함한다.

## 파일 위치

예시 파일:

```text
results/output/scene_g1_motion_dataset/climbing_down_stairs_001/gv/climbing_down_stairs_001_scene_g1_motion.npz
```

생성 명령:

```bash
conda run -n pyroki python scripts/export_scene_g1_motion_dataset.py \
  --seq-name climbing_down_stairs_001 \
  --hmr-type gv \
  --retarget-mode simple \
  --force
```

## 좌표계

모든 위치 데이터는 CRISP post_scene 기준 world frame에 정렬되어 있다.

```text
coordinate_frame = world_z_up_scene_grounded_gravity_down_negative_z
gravity_vector_world = [0, 0, -9.81]
```

즉 `+Z`가 위쪽이고, scene mesh는 지면/최저점이 `z=0`이 되도록 정렬되어 있다. `world_rotation`과 `shared_translation`은 CRISP 원본 scene을 이 좌표계로 변환할 때 사용된 값이다.

## Scene Mesh

Scene은 heightmap이 아니라 triangle mesh로 저장된다.

```text
scene_mesh_vertices: (V, 3) float32
scene_mesh_faces: (F, 3) int32
scene_mesh_bounds: (2, 3) float32
```

`scene_mesh_vertices`는 world 좌표계의 vertex 배열이고, `scene_mesh_faces`는 vertex index triplet이다. simulator에서는 이 둘로 collision mesh 또는 visual mesh를 구성하면 된다. 원본 파일 경로는 추적용 metadata로만 저장된다.

```text
scene_mesh_source: scalar string
scene_urdf_source: scalar string
```

## G1 Motion

G1은 Unitree G1 29-DoF 모델 기준이다.

```text
robot_name: unitree_g1_29dof
g1_dof_names: (29,) string
g1_link_names: (40,) string
```

프레임별 motion은 다음 배열로 저장된다.

```text
g1_root_pos: (T, 3) float32
g1_root_quat_wxyz: (T, 4) float32
g1_dof_pos: (T, 29) float32
g1_dof_vel: (T, 29) float32
g1_root_lin_vel: (T, 3) float32
g1_qpos: (T, 36) float32
```

`g1_qpos`는 convenience 배열이며 `[root_pos(3), root_quat_wxyz(4), dof_pos(29)]` 순서다. 관절 순서는 반드시 `g1_dof_names`를 기준으로 해석해야 한다.

## 시간 정보

```text
fps: scalar float32
dt: scalar float32
frame_count: scalar int32
```

현재 예시 sequence는 `fps=30`, `frame_count=217`이다. simulator에서 rollout timestep을 고정하려면 `dt = 1 / fps`를 사용한다.

## 로드 예시

```python
import numpy as np

data = np.load("climbing_down_stairs_001_scene_g1_motion.npz", allow_pickle=True)

vertices = data["scene_mesh_vertices"]
faces = data["scene_mesh_faces"]
root_pos = data["g1_root_pos"]
root_quat = data["g1_root_quat_wxyz"]
dof_pos = data["g1_dof_pos"]
dof_names = data["g1_dof_names"]
fps = float(data["fps"])
```

## 주의사항

이 파일은 학습용 portable dataset이므로 human SMPL motion, PyRoki optimization 중간값, heightmap은 포함하지 않는다. terrain-aware tracking에서 heightmap이 필요하면 학습 프로젝트 안에서 `scene_mesh_vertices`와 `scene_mesh_faces`로부터 simulator 요구사항에 맞게 별도로 생성하는 것이 좋다.
