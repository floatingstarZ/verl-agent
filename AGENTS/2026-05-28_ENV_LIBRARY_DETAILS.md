# 2026-05-28 Environment Library Details

Generated at: 
- 2026-05-28T15:18:30+00:00

Workspace:
- /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent

## Runtime Summary

```text
python=3.11.11
executable=/opt/conda/bin/python
platform=Linux-5.15.0-136-generic-x86_64-with-glibc2.35
torch=2.9.1+cu128
torch_cuda=12.8
cuda_available=True
cuda_device_count=4
```

## GPU / Driver

- 0, NVIDIA H100 80GB HBM3, 575.57.08, 81559 MiB
- 1, NVIDIA H100 80GB HBM3, 575.57.08, 81559 MiB
- 2, NVIDIA H100 80GB HBM3, 575.57.08, 81559 MiB
- 3, NVIDIA H100 80GB HBM3, 575.57.08, 81559 MiB

## Key Libraries

| Library | Version |
|---|---|
| `numpy` | `1.26.4` |
| `pandas` | `2.3.3` |
| `scipy` | `1.16.3` |
| `matplotlib` | `3.10.7` |
| `seaborn` | `not-installed` |
| `scikit-learn` | `1.7.2` |
| `pyarrow` | `22.0.0` |
| `datasets` | `4.0.0` |
| `transformers` | `4.57.1` |
| `tokenizers` | `0.22.1` |
| `accelerate` | `1.11.0` |
| `peft` | `0.17.1` |
| `trl` | `0.9.6` |
| `vllm` | `0.15.0` |
| `sglang` | `not-installed` |
| `ray` | `2.53.0` |
| `hydra-core` | `1.3.2` |
| `omegaconf` | `2.3.0` |
| `wandb` | `0.25.1` |
| `flash-attn` | `2.8.3` |
| `flashinfer-python` | `0.6.1` |
| `xformers` | `not-installed` |
| `deepspeed` | `0.16.9` |
| `megatron-core` | `0.13.1` |
| `tensordict` | `0.10.0` |
| `einops` | `0.8.1` |
| `sentencepiece` | `0.2.1` |
| `protobuf` | `6.33.0` |
| `pydantic` | `2.12.5` |
| `fastapi` | `0.121.0` |
| `uvicorn` | `0.38.0` |
| `gradio` | `5.45.0` |
| `openai` | `2.16.0` |
| `gym` | `not-installed` |
| `gymnasium` | `not-installed` |
| `jnius` | `not-installed` |
| `spacy` | `not-installed` |
| `beautifulsoup4` | `4.12.3` |
| `lxml` | `not-installed` |
| `pytest` | `not-installed` |

## Full pip freeze (all installed Python packages)

Total packages: `336`

```text
absl-py==2.4.0
accelerate==1.11.0
aiofiles==24.1.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.3
aiohttp-cors==0.8.1
aiosignal==1.4.0
annotated-doc==0.0.3
annotated-types==0.7.0
anthropic==0.77.1
antlr4-python3-runtime==4.9.3
anyio==4.11.0
apache-tvm-ffi==0.1.8.post2
apex==0.1
archspec==0.2.5
astor==0.8.1
asttokens==3.0.0
astunparse==1.6.3
attrs==25.1.0
audioread==3.1.0
av==16.0.1
beautifulsoup4==4.12.3
blake3==1.0.8
boltons==24.0.0
Brotli==1.1.0
cachetools==7.0.0
cbor2==5.8.0
certifi==2024.12.14
cffi==2.0.0
cfgv==3.5.0
chardet==5.2.0
charset-normalizer==3.4.1
click==8.1.8
cloudpickle==3.1.2
cmake==3.31.4
codetiming==1.4.0
colorama==0.4.6
colorful==0.5.8
compressed-tensors==0.13.0
conda==25.1.0
conda-build==25.1.1
conda_index==0.5.0
conda-libmamba-solver==25.1.1
conda-package-handling==2.4.0
conda_package_streaming==0.11.0
contourpy==1.3.3
cryptography==46.0.4
cuda-bindings==13.1.1
cuda-pathfinder==1.3.3
cuda-python==13.1.1
cupy-cuda12x==13.6.0
cycler==0.12.1
datasets==4.0.0
decorator==5.1.1
deepspeed==0.16.9
depyf==0.20.0
dill==0.3.8
diskcache==5.6.3
distlib==0.4.0
distro==1.9.0
dnspython==2.7.0
docstring_parser==0.17.0
einops==0.8.1
email-validator==2.3.0
exceptiongroup==1.2.2
executing==2.1.0
expecttest==0.3.0
fastapi==0.121.0
fastapi-cli==0.0.20
fastapi-cloud-cli==0.11.0
fastar==0.8.0
fastrlock==0.8.3
ffmpy==0.6.4
filelock==3.25.2
fire==0.7.1
flash_attn==2.8.3
flashinfer-python==0.6.1
fonttools==4.60.1
frozendict==2.4.6
frozenlist==1.8.0
fsspec==2024.12.0
gguf==0.17.1
gitdb==4.0.12
GitPython==3.1.46
google-api-core==2.30.0
google-auth==2.49.1
googleapis-common-protos==1.73.0
gradio==5.45.0
gradio_client==1.13.0
groovy==0.1.2
grpcio==1.76.0
grpcio-reflection==1.76.0
h11==0.16.0
h2==4.1.0
hf_transfer==0.1.9
hf-xet==1.4.2
hjson==3.1.0
hpack==4.0.0
httpcore==1.0.9
httptools==0.7.1
httpx==0.28.1
httpx-sse==0.4.3
huggingface-hub==0.36.0
hydra-core==1.3.2
hyperframe==6.0.1
hypothesis==6.124.7
identify==2.6.18
idna==3.10
ijson==3.4.0.post0
importlib_metadata==8.7.1
importlib_resources==6.5.2
interegular==0.3.3
ipython==8.31.0
jedi==0.19.2
jieba==0.42.1
Jinja2==3.1.5
jiter==0.13.0
jmespath==1.1.0
joblib==1.5.2
jsonpatch==1.33
jsonpointer==3.0.0
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
kiwisolver==1.4.9
lark==1.2.2
latex2sympy2_extended==1.11.0
lazy_loader==0.4
libarchive-c==5.1
libmambapy==2.0.5
librosa==0.11.0
lief==0.14.1
liger_kernel==0.6.3
lintrunner==0.12.7
llamafactory==0.9.4.dev0
llguidance==1.3.0
llvmlite==0.44.0
lm-format-enforcer==0.11.3
loguru==0.7.3
Markdown==3.10.2
markdown-it-py==4.0.0
MarkupSafe==3.0.2
math-verify==0.9.0
matplotlib==3.10.7
matplotlib-inline==0.1.7
mbridge==0.15.1
mcp==1.26.0
mdurl==0.1.2
megatron-core==0.13.1
menuinst==2.2.0
mistral_common==1.9.0
ml_dtypes==0.5.4
model-hosting-container-standards==0.1.13
modelscope==1.31.0
more-itertools==10.6.0
mpmath==1.3.0
msgpack==1.1.2
msgspec==0.20.0
multidict==6.7.0
multiprocess==0.70.16
networkx==3.4.2
ninja==1.11.1.3
nltk==3.9.2
nodeenv==1.10.0
numba==0.61.2
numpy==1.26.4
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cudnn-frontend==1.18.0
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-cutlass-dsl==4.3.5
nvidia-ml-py==13.590.48
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.3.20
nvidia-nvtx-cu12==12.8.90
omegaconf==2.3.0
onnx==1.20.1
onnx-ir==0.2.0
onnxscript==0.3.1
openai==2.16.0
openai-harmony==0.0.8
opencensus==0.11.4
opencensus-context==0.1.3
opencv-python==4.13.0.90
opencv-python-headless==4.13.0.90
opentelemetry-api==1.40.0
opentelemetry-exporter-prometheus==0.61b0
opentelemetry-proto==1.40.0
opentelemetry-sdk==1.40.0
opentelemetry-semantic-conventions==0.61b0
optree==0.14.0
orjson==3.11.4
outlines_core==0.2.11
packaging==25.0
pandas==2.3.3
parso==0.8.4
partial-json-parser==0.2.1.1.post7
peft==0.17.1
pexpect==4.9.0
pickleshare==0.7.5
pillow==11.0.0
pip==25.3
pkginfo==1.12.0
pkgutil_resolve_name==1.3.10
platformdirs==4.3.6
pluggy==1.5.0
pooch==1.8.2
pre_commit==4.5.1
prometheus_client==0.24.1
prometheus-fastapi-instrumentator==7.1.0
prompt_toolkit==3.0.50
propcache==0.4.1
proto-plus==1.27.1
protobuf==6.33.0
psutil==6.1.1
ptyprocess==0.7.0
pure_eval==0.2.3
py-cpuinfo==9.0.0
py-spy==0.4.1
pyarrow==22.0.0
pyasn1==0.6.3
pyasn1_modules==0.4.2
pybase64==1.4.3
pybind11==3.0.2
pycosat==0.6.6
pycountry==24.6.1
pycparser==2.22
pydantic==2.12.5
pydantic_core==2.41.5
pydantic-extra-types==2.11.0
pydantic-settings==2.12.0
pydub==0.25.1
Pygments==2.19.1
PyJWT==2.11.0
pylatexenc==2.10
pyparsing==3.2.5
PySocks==1.7.1
python-dateutil==2.9.0.post0
python-discovery==1.2.0
python-dotenv==1.2.1
python-etcd==0.4.5
python-json-logger==4.0.0
python-multipart==0.0.20
pytz==2024.2
pyvers==0.1.0
PyYAML==6.0.2
pyzmq==27.1.0
ray==2.53.0
referencing==0.36.2
regex==2025.11.3
requests==2.32.3
rich==14.2.0
rich-toolkit==0.18.1
rignore==0.7.6
rouge-chinese==1.0.3
rpds-py==0.22.3
ruamel.yaml==0.18.10
ruamel.yaml.clib==0.2.8
ruff==0.14.3
safehttpx==0.1.7
safetensors==0.5.3
scikit-learn==1.7.2
scipy==1.16.3
semantic-version==2.10.0
sentencepiece==0.2.1
sentry-sdk==2.51.0
setproctitle==1.3.7
setuptools==80.9.0
shellingham==1.5.4
shtab==1.7.2
six==1.17.0
smart_open==7.5.1
smmap==5.0.3
sniffio==1.3.1
sortedcontainers==2.4.0
soundfile==0.13.1
soupsieve==2.5
soxr==1.0.0
sse-starlette==3.0.3
stack_data==0.6.3
starlette==0.49.3
supervisor==4.3.0
sympy==1.14.0
tabulate==0.9.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
tensorboardX==2.6.4
tensordict==0.10.0
termcolor==3.2.0
threadpoolctl==3.6.0
tiktoken==0.12.0
tokenizers==0.22.1
tomlkit==0.13.3
torch==2.9.1
torchaudio==2.9.1
torchdata==0.11.0
torchelastic==0.2.2
torchvision==0.24.1
tqdm==4.67.1
traitlets==5.14.3
transformer_engine==2.6.0+c90a720
transformers==4.57.1
triton==3.5.1
trl==0.9.6
truststore==0.10.0
typer==0.20.0
types-dataclasses==0.6.6
typing_extensions==4.15.0
typing-inspection==0.4.2
tyro==0.8.14
tzdata==2025.2
urllib3==2.3.0
uvicorn==0.38.0
uvloop==0.22.1
virtualenv==21.2.0
vllm==0.15.0
wandb==0.25.1
watchfiles==1.1.1
wcwidth==0.2.13
websockets==15.0.1
Werkzeug==3.1.6
wheel==0.45.1
wrapt==2.1.2
xgrammar==0.1.29
xxhash==3.6.0
yarl==1.22.0
zipp==3.21.0
zstandard==0.23.0
```
