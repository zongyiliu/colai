# The RWKV Language Model (RWKV模型—)

## RWKV: Parallelizable RNN with Transformer-level LLM Performance (pronounced as "RwaKuv", from 4 major params: R W K V)
### 介绍
RWKV is an RNN with Transformer-level LLM performance, which can also be directly trained like a GPT transformer (parallelizable). And it's 100% attention-free. You only need the hidden state at position t to compute the state at position t+1. You can use the "GPT" mode to quickly compute the hidden state for the "RNN" mode.

So it's combining the best of RNN and transformer - **great performance, fast inference, saves VRAM, fast training, "infinite" ctx_len, and free sentence embedding** (using the final hidden state).

**Raven 14B** (finetuned on Alpaca+ShareGPT+...) Demo: https://huggingface.co/spaces/BlinkDL/ChatRWKV-gradio

**Raven 7B** (finetuned on Alpaca+ShareGPT+...) Demo: https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B

**ChatRWKV:** with "stream" and "split" strategies and INT8. **3G VRAM is enough to run RWKV 14B :)** https://github.com/BlinkDL/ChatRWKV

**Download RWKV-4 0.1/0.4/1.5/3/7/14B weights**: https://huggingface.co/BlinkDL

**RWKV pip package**: https://pypi.org/project/rwkv/

```python
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # if '1' then use CUDA kernel for seq mode (much faster)
from rwkv.model import RWKV                         # pip install rwkv
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16')

out, state = model.forward([187, 510, 1563, 310, 247], None)   # use 20B_tokenizer.json
print(out.detach().cpu().numpy())                   # get logits
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)           # RNN has state (use deepcopy if you want to clone it)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())                   # same result as above
```
**Cool Community RWKV Projects (check them!)**:

https://github.com/saharNooby/rwkv.cpp fast i4 i8 fp16 fp32 CPU inference using [ggml](https://github.com/ggerganov/ggml)

https://github.com/harrisonvanderbyl/rwkv-cpp-cuda fast windows/linux & cuda/rocm/vulkan GPU inference (no need for python & pytorch)

https://github.com/Blealtan/RWKV-LM-LoRA LoRA fine-tuning

https://github.com/josStorer/RWKV-Runner cool GUI

More RWKV projects: https://github.com/search?o=desc&q=rwkv&s=updated&type=Repositories
