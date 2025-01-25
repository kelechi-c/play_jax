ls /dev/accel*

sudo apt-get update -y -qq
sudo apt-get upgrade -y -qq
sudo apt-get install -y -qq golang neofetch zsh byobu
sudo snap install neofetch btop
sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y -qq python3.12-full python3.12-dev python3.12-venv

pip install -U pip
mkdir playjax && cd playjax
python3.12 -m venv ~/.venv
source .venv/bin/activate

pip install -U wheel
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax datasets numpy tqdm jax-smi wandb transformers

python -c "import jax; print('testing tpus'); print(f'JAX recognized devices:\n local device count = {jax.local_device_count()} \n {jax.local_devices()}')"