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
    ),
    network_file_systems={"/content": volume},
    gpu="A10G",
    timeout=60000,
)
async def run():
    os.system(f"git clone -b dev https://github.com/camenduru/One-2-3-45-hf /content/test")
    os.chdir(f"/content/test")
    os.environ['HF_HOME'] = '/content/cache/huggingface'
    os.system(f"pip install taming-transformers-rom1504")
    os.system(f"python app.py")

@stub.local_entrypoint()
def main():
    run.remote()