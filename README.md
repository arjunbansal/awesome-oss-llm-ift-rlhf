# Awesome Open Source LLM + IFT (+RLHF)

This is a collection of open source implementations of LLMs with IFT and RLHF that are striving to get to ChatGPT level of performance. Some observations that might help with trying out the demos quickly for yourself:

* Most of the Colab notebooks require a Colab Pro account ($9.99/month) to get Premium GPU access
* Quantized GPT4All can run on a laptop CPU thanks to [llama.cpp](https://github.com/ggerganov/llama.cpp)
* LoRA models are fine-tuneable on consumer hardware (e.g. RTX4090) whereas the non-LoRA models seem to require 8-10 hours on 8xA100 systems (costing \<$100 of compute time)

## Models

| Name | Base model | IFT | IFT data | RLHF | [LoRA](https://arxiv.org/abs/2106.09685) | Quantization | Commercial Use|Links|
|------|------------|-----|----------|:----:|:----:|--------------|------|------|
|[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)|Llama 7B<br>Llama 13B| ✅ | Alpaca (davinci-003) <br>gpt-4| ❌|❌ |❌|❌| [Alpaca model](https://huggingface.co/chavinlo/alpaca-native)<br>[GPT-4 model](https://huggingface.co/chavinlo/gpt4-x-alpaca)|
|Alpaca+LORA|Llama 7B    | ✅  | Alpaca (davinci-003) Cleaned| ❌|✅ |❌|❌| [Spaces](https://huggingface.co/spaces/tloen/alpaca-lora)<br>[Github](https://github.com/tloen/alpaca-lora)|
|[GPT4All](https://s3.amazonaws.com/static.nomic.ai/gpt4all/2023_GPT4All_Technical_Report.pdf)|Llama 7B| ✅  | gpt-3.5| ❌|✅|✅|❌|[Github](https://github.com/nomic-ai/gpt4all)|
|[Instruct GPT-J+LoRA](https://twitter.com/aicrumb/status/1638630904569511938)|GPT-J-6B | ✅ |Alpaca (davinci-003)| ❌|✅ |❌|❌| [Colab](https://colab.research.google.com/github/aicrumb/notebook-hosting/blob/main/Instruct_GPT_J_Gradio_Demo.ipynb)<br>[Model](https://huggingface.co/crumb/Instruct-GPT-J)|
|[Dolly](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html)|GPT-J-6B | ✅ | Alpaca (davinci-003)| ❌|❌ |❌ |❌ | [Model](https://huggingface.co/databricks/dolly-v1-6b)<br>[Github](https://github.com/databrickslabs/dolly)|
|[Dolly+LoRA](https://twitter.com/Sam_Witteveen/status/1639947728762593280)|GPT-J-6B | ✅ | Alpaca (davinci-003) Cleaned| ❌|✅ |❌ |❌ | [Colab](https://colab.research.google.com/drive/1O1JjyGaC300BgSJoUbru6LuWAzRzEqCz?usp=sharing)|
|[OpenChatKit](https://www.together.xyz/blog/openchatkit-016)|Pythia 7B<br>GPT-NeoXT-20B | ✅  | [LAION OIG](https://huggingface.co/datasets/laion/OIG)| ✅|❌|✅ |✅ | [Spaces](https://huggingface.co/spaces/togethercomputer/OpenChatKit)<br>[Github](https://github.com/togethercomputer/OpenChatKit)|
|ColossalChat|Llama-7B | ✅  | | ✅|✅  |✅  |❌ | [Demo](https://chat.colossalai.org/)<br>[Github](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)|
|[TRL-PEFT](https://huggingface.co/blog/trl-peft)|   | ✅  | | ✅|✅ |✅  ||code only, no model |

## Benchmarks

Several of the above models are in the process of formal benchmarking using [HELM](https://crfm.stanford.edu/helm/latest/). We'll update this space as these numbers are released.