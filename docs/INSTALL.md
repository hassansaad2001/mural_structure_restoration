# Installation and running (Ubuntu + Docker)

Everything that needs PyTorch and a GPU (training, testing, evaluation and inference servers) runs in a
**Docker container** built from [`docker/Dockerfile`](../docker/Dockerfile). Its base image is
`pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel`, with pinned package versions.
The two **GUIs** run directly on the host and only need Python with tkinter. They talk to the servers in the container over TCP.

```
 Ubuntu host                                   Docker container "msr"  (GPU)
 ─────────────────────────────                 ──────────────────────────────────────────
 scripts/eval_gui.sh       ── TCP 9998 ──▶     evaluation/eval_server.py
 scripts/inference_gui.sh  ── TCP 9999 ──▶     inference/inference_server.py
                                               train_tsr.py, generate_edges.py, tests
 <parent folder of the repo>   ═══ mounted at ═══▶  /master
```

The same steps work on **native Ubuntu 22.04 / 24.04** and inside **WSL 2** (see the end of this page).

---

## 1. One-time host setup

### 1.1 NVIDIA driver

```bash
sudo ubuntu-drivers install
```

Reboot, then check that `nvidia-smi` lists your GPUs. The CUDA 11.6 image runs on any current driver.

### 1.2 Docker Engine

Install it:

```bash
curl -fsSL https://get.docker.com | sudo sh
```

Let your user run Docker without `sudo` (log out and back in afterwards):

```bash
sudo usermod -aG docker $USER
```

### 1.3 NVIDIA Container Toolkit (GPUs inside containers)

1. Add NVIDIA's signing key:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

2. Add NVIDIA's package repository:

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

3. Install the toolkit:

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

4. Connect it to Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

5. Check that a container can see the GPUs:

```bash
docker run --rm --gpus all pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel nvidia-smi
```

### 1.4 Python + tkinter for the GUIs (host)

1. Install the system packages:

```bash
sudo apt install -y python3-venv python3-tk
```

2. From the repository root, create the small GUI environment in `.venv-gui/`:

```bash
bash scripts/setup_gui_env.sh
```

---

## 2. Get the code and the data

```bash
git clone <your GitHub URL> mural_structure_restoration
```

`docker/run.sh` mounts the **parent folder** of the repository at `/master`. The simplest layout puts the dataset next to the repository:

```
~/MAT/                               ->  /master
├── mural_structure_restoration/     ->  /master/mural_structure_restoration   (this repo)
├── DhMurals-inpainting-dataset/     ->  /master/DhMurals-inpainting-dataset   (DATA_ROOT default)
└── outputs/                             (optional, for runs)
```

Elsewhere you have two options:

- **Mount the dataset folder separately:** set `MSR_DATA=/path/to/dataset` when you start the container; it appears at `/data`. Then run training with `DATA_ROOT=/data`.
- **Mount a different parent folder:** set `MSR_MOUNT=/some/folder`.

## 3. Start the container

```bash
bash docker/run.sh
```

The first start builds the image `mural-structure:latest`, which takes a few minutes. Later starts reuse it.
The container keeps running in the background. It is named `msr`, and its working directory is the repository.

| Variable | Default | Meaning |
|---|---|---|
| `MSR_CONTAINER` | `msr` | container name |
| `MSR_GPUS` | `all` | e.g. `'"device=0,1"'`, or `none` |
| `MSR_SHM` | `32g` | shared memory for DataLoader workers |
| `MSR_MOUNT` | parent of the repo | host folder mounted at `/master` |
| `MSR_DATA` | – | extra host folder mounted at `/data` |
| `MSR_EVAL_PORT` / `MSR_INFER_PORT` / `MSR_TB_PORT` | 9998 / 9999 / 6006 | host ports |
| `MSR_AS_USER=1` | off | run as your user, so files written to the mount are not owned by root |
| `MSR_NETWORK=host` | off | use the host network (servers listen directly on the host's ports); fixes very slow downloads inside the container |

Open a shell in the container:

```bash
docker exec -it msr bash
```

## 4. Train, test, evaluate

All commands in this section run **inside the container**, from the repository folder.

| Task | Command |
|---|---|
| Train the final model | `bash scripts/train_example.sh` |
| Train the baselines | `bash scripts/train_edgeconnect.sh`, `bash scripts/train_sgtb_srn.sh`, `bash scripts/train_zits_tsr.sh` |
| All training options | `python3 train_tsr.py --help` (see [TRAINING.md](TRAINING.md), [BASELINES.md](BASELINES.md)) |
| Predict a test folder | `bash scripts/test_example.sh <ckpt> <images> <edges> <masks> <out>` |
| Headless evaluation | `python3 evaluation/eval_server.py --headless --settings eval_settings.json` |
| Check the installation | `python3 -m pytest tests -q` (CPU only, about 2–3 min) |

- **Long trainings:** keep running after you close the terminal with `nohup ... > train.log 2>&1 &`. You can also start them from the host with `docker exec -d msr bash -c "..."`.
- **Choosing GPUs:** prefix a command with `CUDA_VISIBLE_DEVICES=0,1`.
- **TensorBoard:** start it from the host, then open http://localhost:6006:

```bash
docker exec -d msr tensorboard --logdir /master/mural_structure_restoration/outputs --host 0.0.0.0 --port 6006
```

## 5. GUIs (on the host)

| GUI | Command | What it does |
|---|---|---|
| Checkpoint evaluation | `bash scripts/eval_gui.sh` | starts `eval_server.py` in the container, opens the GUI, stops the server when you close it |
| Interactive inference | `bash scripts/inference_gui.sh <checkpoint.pth>` | same, with `inference_server.py` loading the checkpoint |

- **Paths:** enter folders in the GUI as normal host paths. The launcher sets the path map `<parent folder>=/master`, so the server finds the same files. Checkpoints and data must therefore lie inside the mounted folder.
- **Using another container:** set `MSR_CONTAINER=<name>`, for example a container you started yourself. It needs the same `/master` mount. Ports 9998/9999 don't have to be published: the launchers then connect to the container's IP address.
- **Remote GPU server:** use `ssh -X user@server` so the GUI window opens on your machine.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing or Docker not restarted (step 1.3) |
| `Bus error` / DataLoader killed | increase `MSR_SHM` and recreate the container (`docker rm -f msr && bash docker/run.sh`) |
| Files in `outputs/` owned by root | `sudo chown -R $USER outputs`, or use `MSR_AS_USER=1` for new containers |
| Downloads inside containers crawl (tens of kB/s), image build takes hours | Docker's bridge network has an MTU problem on this machine (common on WSL). `run.sh` already builds with `--network=host`; also start the container with `MSR_NETWORK=host`, which affects for example FID's Inception download |
| `port is already allocated` | another container publishes 9998/9999/6006: set `MSR_EVAL_PORT` etc. |
| GUI: `No module named tkinter` | `sudo apt install python3-tk` and rerun `scripts/setup_gui_env.sh` |
| GUI: `couldn't connect to display` | no desktop session: use `ssh -X`, or run on the machine's desktop |
| First training start is slow | the custom CUDA ops in `torch_utils/ops` are compiled once per container |

## 7. WSL 2 (Windows)

The same steps work inside an Ubuntu WSL 2 distribution:

- **GPU driver:** install the NVIDIA driver on **Windows**, not inside WSL, so skip step 1.1. `nvidia-smi` then works inside WSL.
- **Docker:** use either Docker Engine installed inside WSL (step 1.2 + 1.3), or Docker Desktop with WSL integration.
- **GUIs:** they need WSLg, which is included in Windows 11. In the evaluation GUI, the *Open results* button opens Windows Explorer.
