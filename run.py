import modal
import os

stub = modal.Stub("test")
volume = modal.NetworkFileSystem.new().persisted("test")

@stub.function(
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        "apt update -y && \
        apt install -y software-properties-common && \
        apt update -y && \
        add-apt-repository -y ppa:git-core/ppa && \
        apt update -y && \
        apt install -y git git-lfs && \
        git --version  && \
        apt install -y aria2 libgl1 libglib2.0-0 wget libsparsehash-dev build-essential clang && \
        pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 && \
        pip install -q xformers==0.0.20 triton==2.0.0 packaging==23.1"
    )
    .pip_install_from_requirements(
        "requirements.txt"
    )
    .pip_install(
        "git+https://github.com/CompVis/taming-transformers.git#egg=taming-transformers",
        "git+https://github.com/NVlabs/nvdiffrast",
        "git+https://github.com/openai/CLIP",
        "git+https://github.com/facebookresearch/segment-anything",
    ),
    network_file_systems={"/content": volume},
    gpu="A10G",
    timeout=60000,
)
async def run():
    import pip
    # pip.main(['install', elev_est_dir])
    # export TORCH_CUDA_ARCH_LIST="7.0;7.2;8.0;8.6"
    # export IABN_FORCE_CUDA=1
    # os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
    # os.environ["IABN_FORCE_CUDA"] = "1"
    # os.environ["FORCE_CUDA"] = "1"
    # pip.main(["install", "inplace_abn"]) 
    # FORCE_CUDA=1 pip install --no-cache-dir git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
    # pip.main(["install", "--no-cache-dir", "git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0"])

    # os.system(f"pip show torch torchvision albumentations opencv-python pudb imageio imageio-ffmpeg pytorch-lightning omegaconf test-tube streamlit einops torch-fidelity transformers kornia webdataset torchmetrics fire gradio diffusers datasets[vision] rich plotly dl_ext loguru matplotlib Pillow PyYAML scikit_image scikit_learn scipy setuptools tensorboardX tqdm transforms3d trimesh yacs gdown onnxruntime  onnx rembg pyhocon icecream PyMCubes ninja")
    # os.system(f"!rm -rf /content/test")
    # os.system(f"git lfs install")
    # os.system(f"git clone -b dev https://github.com/camenduru/One-2-3-45-hf /content/test")
    # os.chdir(f"/content/test")
    # os.system(f"ls -al")
    os.system(f"pip install git+https://github.com/mit-han-lab/torchsparse@v1.4.0")
    os.system(f"pip show torchsparse")
    # os.environ['HF_HOME'] = '/content/cache/huggingface'
    # os.system(f"python app.py")

@stub.local_entrypoint()
def main():
    run.remote()