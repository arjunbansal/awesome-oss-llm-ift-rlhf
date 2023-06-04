# Awesome Open Source LLM + IFT (+RLHF)

This is a collection of open source implementations of LLMs with IFT and RLHF that are striving to get to ChatGPT level of performance. Some observations that might help with trying out the demos quickly for yourself:

* Most of the Colab notebooks require a Colab Pro account ($9.99/month) to get Premium GPU access
* Quantized GPT4All can run on a laptop CPU thanks to [llama.cpp](https://github.com/ggerganov/llama.cpp)
* LoRA models are fine-tuneable on consumer hardware (e.g. RTX4090) whereas the non-LoRA models seem to require 8-10 hours on 8xA100 systems (costing \<$100 of compute time)

## Commercial-use models

| Name | Base model | IFT | IFT data | RLHF | [LoRA](https://arxiv.org/abs/2106.09685) | Quantization | Commercial Use|Links|
|------|------------|:--:|----------|:----:|:----:|:-----:|:------:|------|
|[Falcon](https://falconllm.tii.ae/)|Falcon-40B instruct | ✅ | [Baize](https://github.com/project-baize/baize-chatbot)| ❌|❌ |❌ |✅ | [Model](https://huggingface.co/tiiuae/falcon-40b-instruct)|
|[MPT](https://www.mosaicml.com/blog/mpt-7b)|MPT-7B instruct | ✅ | dolly-15k<br>Anthropic| ❌|❌ |❌ |✅ | [Spaces](https://huggingface.co/spaces/mosaicml/mpt-7b-instruct)|
|[Dolly 2.0](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)|Pythia-12B | ✅ | dolly-15k| ❌|❌ |❌ |✅ | [Model](https://huggingface.co/databricks/dolly-v2-12b)<br>[Github](https://github.com/databrickslabs/dolly)|
|[OpenChatKit](https://www.together.xyz/blog/openchatkit-016)|Pythia 7B<br>GPT-NeoXT-20B | ✅  | [LAION OIG](https://huggingface.co/datasets/laion/OIG)| ❌|❌|✅ |✅ | [Spaces](https://huggingface.co/spaces/togethercomputer/OpenChatKit)<br>[Github](https://github.com/togethercomputer/OpenChatKit)|
|[Open Assistant](https://www.ykilcher.com/OA_Paper_2023_04_15.pdf)|Pythia 12B<br> | ✅  | [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1)|✅ |❌|❌ |✅ | [Demo](https://open-assistant.io/)<br>[Model](https://huggingface.co/OpenAssistant)<br>[Github](https://github.com/LAION-AI/Open-Assistant)|

## Non commercial-use models

| Name | Base model | IFT | IFT data | RLHF | [LoRA](https://arxiv.org/abs/2106.09685) | Quantization | Commercial Use|Links|
|------|------------|:--:|----------|:----:|:----:|:-----:|:------:|------|
|[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)|Llama 7B<br>Llama 13B| ✅ | Alpaca (davinci-003) <br>gpt-4| ❌|❌ |❌|❌| [Alpaca model](https://huggingface.co/chavinlo/alpaca-native)<br>[GPT-4 model](https://huggingface.co/chavinlo/gpt4-x-alpaca)|
|[Vicuna](https://vicuna.lmsys.org/)|Llama 13B|✅ |ShareGPT|❌|❌ |❌|❌|[Demo](https://chat.lmsys.org/)<br>[Github](https://github.com/lm-sys/FastChat/#vicuna-weights)|
|[Koala<br>EasyLM](https://bair.berkeley.edu/blog/2023/04/03/koala/)|Llama 13B | ✅  |Alpaca<br>ShareGPT<br>HC3<br>LAION OIG<br>Anthropic<br>WebGPT<br>Summaries |❌|❌ |❌|❌|[Demo](https://chat.lmsys.org/?model=koala-13b)<br>[Github](https://github.com/young-geng/EasyLM)|
|Alpaca+LORA|Llama 7B    | ✅  | Alpaca (davinci-003) Cleaned| ❌|✅ |❌|❌| [Spaces](https://huggingface.co/spaces/tloen/alpaca-lora)<br>[Github](https://github.com/tloen/alpaca-lora)|
|[Baize](https://arxiv.org/pdf/2304.01196.pdf)|Llama 7B<br>Llama 13B<br>Llama 30B    | ✅  | gpt-3.5-turbo| ❌|✅ |❌|❌| [Spaces](https://huggingface.co/spaces/project-baize/baize-lora-7B)<br>[Github](https://github.com/project-baize/baize)|
|[GPT4All](https://s3.amazonaws.com/static.nomic.ai/gpt4all/2023_GPT4All_Technical_Report.pdf)|Llama 7B| ✅  | gpt-3.5| ❌|✅|✅|❌|[Github](https://github.com/nomic-ai/gpt4all)|
|[Instruct GPT-J+LoRA](https://twitter.com/aicrumb/status/1638630904569511938)|GPT-J-6B | ✅ |Alpaca (davinci-003)| ❌|✅ |❌|❌| [Colab](https://colab.research.google.com/github/aicrumb/notebook-hosting/blob/main/Instruct_GPT_J_Gradio_Demo.ipynb)<br>[Model](https://huggingface.co/crumb/Instruct-GPT-J)|
|[Dolly](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html)|GPT-J-6B | ✅ | Alpaca (davinci-003)| ❌|❌ |❌ |❌ | [Model](https://huggingface.co/databricks/dolly-v1-6b)<br>[Github](https://github.com/databrickslabs/dolly)|
|[Dolly+LoRA](https://twitter.com/Sam_Witteveen/status/1639947728762593280)|GPT-J-6B | ✅ | Alpaca (davinci-003) Cleaned| ❌|✅ |❌ |❌ | [Colab](https://colab.research.google.com/drive/1O1JjyGaC300BgSJoUbru6LuWAzRzEqCz?usp=sharing)|
|ColossalChat|Llama 7B | ✅  | | ✅|✅  |✅  |❌ | [Demo](https://chat.colossalai.org/)<br>[Github](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)|
|ChatRMKV|RMKV<br>RNN based   | ✅ | Alpaca| ❌|❌ |✅  |❌|[Spaces](https://huggingface.co/spaces/BlinkDL/ChatRWKV-gradio)<br>[Github](https://github.com/BlinkDL/ChatRWKV)|
|[StableLM](https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models)|StableLM-base| ✅ | Alpaca, GPT4All, Dolly, ShareGPT, and HH| ❌|❌ |❌|❌| [Spaces](https://huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat)<br>[Github](https://github.com/stability-AI/stableLM/)|
|[MPT](https://www.mosaicml.com/blog/mpt-7b)|MPT-7B chat | ✅ | [Anthropic](https://huggingface.co/datasets/sam-mosaic/hhrlhf_evol_chatml)<br>[Vicuna](https://huggingface.co/datasets/sam-mosaic/vicuna_alpaca_hc3_chatml)<br>Alpaca<br>HC3<br>Evol-instruct| ❌|❌ |❌ |❌ | [Spaces](https://huggingface.co/spaces/mosaicml/mpt-7b-chat)|


## Code only

| Name | Base model | IFT | IFT data | RLHF | [LoRA](https://arxiv.org/abs/2106.09685) | Quantization | Commercial Use|Links|
|------|------------|:--:|----------|:----:|:----:|:-----:|:------:|------|
|[TRL-PEFT](https://huggingface.co/blog/trl-peft)|   | ✅  | | ✅|✅ |✅  ||code only, no model |
|[DeepSpeed Chat](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat)|OPT  | ✅  | | ✅|✅ |✅  ||code only, no model |


## Benchmarks

The following resources maintain active benchmarks of the above and similar models:
- [HELM](https://crfm.stanford.edu/helm/latest/)
- [Huggingface Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [LMSys Elo Leaderboard](https://lmsys.org/blog/2023-05-25-leaderboard/)
