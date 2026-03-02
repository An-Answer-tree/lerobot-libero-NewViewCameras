conda create -n libero python=3.10
conda activate libero

conda install nvidia::cuda-toolkit==13.0.0
uv pip install -r requirements.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

git clone --branch v1.4.0 https://github.com/ARISE-Initiative/robosuite.git
cd ../robosuite
CC=/usr/bin/gcc uv pip install -e .
cd ../LIBERO
uv pip install -e .
