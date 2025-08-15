## Setup 
conda create --name nemo python==3.10.12
conda activate nemo

git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout @${REF:-'main'}
pip install '.[all]'

pip install -e '.[asr]' # or pip install 
->> Chỉ nên cài asr 
->> cài từ local
->> cài với mode develop


## Dataset 
- bám theo file code mẫu vpb và vivos_dataset 

## Model

### Tokenizer 
