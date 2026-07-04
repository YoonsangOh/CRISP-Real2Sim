# 로컬 셋업 및 저장공간 가이드

이 저장소는 CRISP-Real2Sim 코드와 실행 스크립트만 추적한다. 모델 weight, SMPL/SMPL-X 파일, 데이터셋, `results/` 산출물, `third_party/` 로컬 clone은 GitHub에 올리지 않는다.

## Clone

```bash
git clone --recursive https://github.com/YoonsangOh/CRISP-Real2Sim.git
cd CRISP-Real2Sim
```

이미 clone한 뒤라면:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## 기본 환경

```bash
bash setups/setup_crisp.sh
conda activate crisp
bash setups/validate_crisp_video_env.sh
```

필요한 body model과 checkpoint는 별도로 받는다.

```bash
cd prep
bash smplme.sh
cd ..
bash setups/fetch_crisp_assets.sh
```

SMPL/SMPL-X는 라이선스가 있는 파일이라 git에 포함하지 않는다. 수동 배치가 필요하면 `prep/README.md`의 경로 규칙을 따른다.

## 출력 위치

CRISP 기본 출력은 다음 아래에 생긴다.

```text
results/init/
results/output/scene/
results/output/post_scene/
logs/
```

이 디렉토리들은 용량이 매우 커질 수 있으므로 `.gitignore`에 포함되어 있다. NAS quota가 빡빡한 서버에서는 저장소를 로컬 디스크에 clone하거나, 결과 디렉토리를 로컬 디스크로 symlink하는 방식을 권장한다.

단일 비디오를 Scene + G1 motion dataset까지 처리할 때는 작업 루트와 로그 위치를 명시할 수 있다.

```bash
GPU_ID=0 \
WORK_ROOT=/local_disk/path/crisp_work \
LOG_DIR=/local_disk/path/crisp_logs \
bash scripts/run_video_to_scene_g1_motion.sh /abs/path/to/video.mp4
```

이 wrapper는 `GPU_ID`를 `CRISP_GPU_IDS`와 `CUDA_VISIBLE_DEVICES`로 전달해, 지정한 GPU 하나만 사용하게 한다.

## PyRoki / G1 Retarget 옵션

`scripts/retarget_crisp_scene_to_g1_pyroki.py`와 관련 export/visualization 스크립트는 PyRoki를 사용한다. PyRoki 소스는 `third_party/pyroki` 아래에 둔다. 이 폴더는 git에 포함하지 않는다.

예시:

```bash
mkdir -p third_party
git clone https://github.com/chungmin99/pyroki.git third_party/pyroki
```

환경 이름은 기본적으로 `pyroki`를 사용한다. 다르면 실행 시 `PYROKI_ENV=<env_name>`을 지정한다.

## Git에 올리지 않는 항목

다음은 재생성하거나 별도로 다운로드해야 하므로 커밋하지 않는다.

```text
results/
logs/
batch_work/
data/
videomimic_*/
third_party/
prep/data/body_models/
prep/*/checkpoints/
*.npy, *.npz, *.pth, *.pt, *.ckpt, *.pkl
```

새 실험을 추가할 때도 결과물 대신 실행 스크립트, 설정 문서, manifest처럼 재현에 필요한 작은 텍스트 파일만 커밋한다.
