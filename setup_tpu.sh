#until gcloud alpha compute tpus tpu-vm create trcnode-1 --project second-modem-441107-i1 --zone us-central2-b --accelerator-type v4-32 --version tpu-vm-base ; do : ; done

ls /dev/accel*

sudo apt-get update -y -qq
sudo apt-get upgrade -y -qq
sudo apt-get install -y -qq golang neofetch zsh byobu

sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y -qq python3.12-full python3.12-dev

pip install -U pip jax-smi uv
# python3.12 -m venv ~/venv
uv venv .venv
source .venv/bin/activate

pip install -U wheel
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

python -c "import jax; print('testing tpus'); print(f'JAX recognized devices:\n local device count = {jax.local_device_count()} \n {jax.local_devices()}')"