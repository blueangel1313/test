import modal
import os

stub = modal.Stub("test")
volume = modal.NetworkFileSystem.new().persisted("test")

@stub.function(
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .run_commands(
        "apt update -y && \
        apt install -y software-properties-common && \
        apt update -y && \
        add-apt-repository -y ppa:git-core/ppa && \
        apt update -y && \
        apt install -y git git-lfs && \
        git --version  && \
        apt install -y aria2 libgl1 libglib2.0-0 wget libsparsehash-dev build-essential clang && \
        pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118"
    )
    .pip_install_from_requirements(
        "requirements.txt"
    )
    .pip_install(
        "git+https://github.com/CompVis/taming-transformers.git#egg=taming-transformers",
        "git+https://github.com/NVlabs/nvdiffrast",
        "git+https://github.com/openai/CLIP",
        "git+https://github.com/facebookresearch/segment-anything",
        "https://github.com/camenduru/One-2-3-45/releases/download/colab/inplace_abn-1.1.0-cp310-cp310-linux_x86_64.whl",
        "https://github.com/camenduru/One-2-3-45/releases/download/colab/torchsparse-1.4.0-cp310-cp310-linux_x86_64.whl",
        "taming-transformers-rom1504",
    ),
    network_file_systems={"/content": volume},
    gpu="A10G",
    timeout=60000,
)
async def run():
    os.environ['HF_HOME'] = '/content/cache/huggingface'
    os.system(f"git lfs install")
    os.system(f"git clone -b dev https://github.com/camenduru/One-2-3-45 /content/One-2-3-45")
    os.chdir(f"/content/One-2-3-45")
    os.system(f"git pull")
    os.system(f"python run.py --img_path /content/One-2-3-45/demo/demo_examples/00_zero123_lysol.png --half_precision")
    os.system(f"ls -la /content/One-2-3-45/exp/00_zero123_lysol")

@stub.local_entrypoint()
def main():
    run.remote()