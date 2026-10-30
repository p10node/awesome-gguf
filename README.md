# Awesome GGUF [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of awesome **GGUF** runtimes, tools, libraries, models, and comprehensive guides for running local Large Language Models (LLMs).

[English](README.md) | [Tiếng Việt 🇻🇳](README.vi.md)

---

## 📑 Table of Contents

- [💡 What is GGUF?](#-what-is-gguf)
  - [The Core Philosophy](#the-core-philosophy)
  - [GGUF vs. GGML vs. Safetensors](#gguf-vs-ggml-vs-safetensors)
- [🎛️ Quantization Demystified](#️-quantization-demystified)
  - [Deconstructing GGUF Filenames](#deconstructing-gguf-filenames)
  - [Quantization Types Overview](#quantization-types-overview)
  - [VRAM & Hardware Sizing Guide](#vram--hardware-sizing-guide)
- [🖥️ Desktop & Turnkey Applications](#️-desktop--turnkey-applications)
- [⚙️ Inference Engines & Serving](#️-inference-engines--serving)
- [📱 Mobile & Edge AI](#-mobile--edge-ai)
- [🛠️ Quantization & Model Tooling](#️-quantization--model-tooling)
- [📚 SDKs & Language Bindings](#-sdks--language-bindings)
- [🌐 Model Hubs & Top Quantizers](#-model-hubs--top-quantizers)
- [📖 Deep Dives, Specs & Articles](#-deep-dives-specs--articles)
- [🤝 Contributing](#-contributing)

---

## 💡 What is GGUF?

**GGUF** (GPT-Generated Unified Format) is the binary format specification designed by [Georgi Gerganov](https://github.com/ggerganov) and the [ggml](https://github.com/ggml-org/ggml) team in August 2023. It serves as the de-facto standard for storing and distributing quantized Large Language Models for local CPU and GPU inference.

### The Core Philosophy

Think of GGUF as the **`.mp3` or `.mkv` of AI models**:
- **Single-File Distribution:** Traditional models (like Hugging Face PyTorch/Safetensors checkpoints) are split across multiple weight shards, configuration JSONs, tokenizer definitions, and special token dictionaries. GGUF packs **the architecture, weights, tokenizer, and metadata into a single `.gguf` file**.
- **Extensible Key-Value Metadata:** Unlike legacy formats where modifying metadata broke parser compatibility, GGUF stores metadata in structured key-value pairs. New model architectures can be introduced without breaking backward compatibility for existing parsers.
- **Instant Loading via `mmap`:** GGUF is designed for direct memory-mapping (`mmap`). Operating systems can map the model file into RAM/VRAM instantly, bypassing slow sequential disk reads.
- **Native Quantization Support:** Built from the ground up to support reduced-precision integer representations (from 1.5-bit to 8-bit), allowing 70B+ parameter models to execute efficiently on consumer-grade hardware.

### GGUF vs. GGML vs. Safetensors

| Feature                    | Hugging Face Safetensors                      | GGML (Legacy / Obsolete)                | GGUF (Current Standard)                       |
|:---------------------------|:----------------------------------------------|:----------------------------------------|:----------------------------------------------|
| **Primary Use-Case**       | Training, Fine-Tuning, Datacenter GPU         | Legacy CPU inference                    | High-performance Local & Edge Inference       |
| **Packaging**              | Multi-file (`.json` configs + `.safetensors`) | Single file                             | **Single self-contained binary file**         |
| **Metadata Flexibility**   | JSON header                                   | Hardcoded struct (brittle)              | **Extensible Key-Value Store**                |
| **Backward Compatibility** | High                                          | Poor (breaking changes across versions) | **High (Forward & Backward compatible)**      |
| **Quantization Variety**   | FP16, BF16, FP8, AWQ, GPTQ                    | Basic Q4_0, Q4_1, Q8_0                  | **K-Quants, I-Matrix (IQ1 to IQ8)**           |
| **Inference Hardware**     | High-VRAM GPUs (CUDA, ROCm)                   | CPU only                                | **CPU (AVX/NEON), Apple Metal, CUDA, Vulkan** |

---

## 🎛️ Quantization Demystified

### Deconstructing GGUF Filenames

A standard GGUF filename encodes model identity, scale, and quantization precision:

```
Llama-3.1-8B-Instruct.Q4_K_M.gguf
│─────────┬─────────│ │───┬───│
          │               └── Quantization Scheme (4-bit, K-Quant, Medium)
          └────────────────── Model Architecture, Parameter Count & Variant
```

- **Bit Level (`Q4`, `Q5`, `Q8`, `IQ3`):** The target precision (e.g., `4` bits per weight vs. 16 bits in original FP16).
- **Quantization Flavor:**
  - `_0` / `_1`: Legacy round-to-nearest quantizations.
  - `_K`: **K-Quants** (K-means block quantization with mixed precision across attention and feed-forward layers).
  - `IQ`: **I-Matrix (Importance Matrix)** quantization, which uses training data activations to preserve crucial weights at extreme low bitrates (1-3 bits).
- **Scale (`_XS`, `_S`, `_M`, `_L`):** Extra Small, Small, Medium, Large. Dictates which internal tensors are kept at higher bit precision.

### Quantization Types Overview

| Type                    | Quality Loss (Perplexity) | Relative Size   | Recommendation / Best For                                      |
|:------------------------|:--------------------------|:----------------|:---------------------------------------------------------------|
| **`Q8_0`**              | Extremely Low (~0%)       | ~55% of FP16    | Near-lossless accuracy; baseline testing.                      |
| **`Q6_K`**              | Negligible (<0.1%)        | ~45% of FP16    | Precision-critical tasks (coding, complex reasoning).          |
| **`Q5_K_M`**            | Very Low (~0.1-0.2%)      | ~38% of FP16    | Balanced choice when RAM/VRAM is available.                    |
| **`Q4_K_M`**            | Low (~0.3-0.5%)           | ~30% of FP16    | 🏆 **The Universal Sweet Spot** (best speed-to-quality ratio). |
| **`Q4_K_S`**            | Moderate                  | ~28% of FP16    | Slightly smaller footprint with minor quality degradation.     |
| **`IQ3_M` / `IQ3_S`**   | Moderate                  | ~23% of FP16    | Best choice for tight VRAM budget; outperforms legacy Q3.      |
| **`IQ2_M` / `IQ2_XXS`** | Noticeable                | ~15-18% of FP16 | Extreme compression to fit 70B models on 16GB-24GB RAM.        |

### VRAM & Hardware Sizing Guide

To estimate the memory needed to run a GGUF model:

$$\text{Estimated RAM/VRAM (GB)} \approx \left(\text{Parameters (B)} \times \frac{\text{Bits}}{8} \times 1.2\right) + \text{KV Cache Overhead (1-2 GB)}$$

*Quick reference for standard setups:*
- **8 GB Unified Memory / VRAM:** Runs **7B / 8B** models at `Q4_K_M`.
- **16 GB Unified Memory / VRAM:** Runs **7B / 8B** at `Q8_0` or **14B** at `Q4_K_M`.
- **24 GB VRAM (RTX 3090 / 4090):** Runs **32B** at `Q4_K_M` or **70B** at `IQ2_XS`.
- **64 GB+ Unified Memory (Mac Studio/Pro):** Runs **70B** at `Q4_K_M` / `Q5_K_M` with full context.

---

## 🖥️ Desktop & Turnkey Applications

User-friendly desktop clients and turnkey applications requiring zero coding setup:

- **[LM Studio](https://lmstudio.ai/)** — Beautiful desktop application to search Hugging Face GGUF models, download, chat, and run a local OpenAI-compatible API server.
- **[Jan.ai](https://github.com/janhq/jan)** — Open-source ChatGPT alternative running 100% locally on your machine with full GGUF support.
- **[Ollama](https://github.com/ollama/ollama)** — CLI tool and background service to get up and running with GGUF models via Modelfiles and REST API.
- **[Open WebUI](https://github.com/open-webui/open-webui)** — Self-hosted feature-packed AI interface with Ollama and OpenAI API compatibility, RAG, and multi-user management.
- **[llamafile](https://github.com/Mozilla-Ocho/llamafile)** — Combines `llama.cpp` and Cosmopolitan Libc into a single multi-gigabyte executable that runs locally on Windows, macOS, Linux, FreeBSD without installation.
- **[Msty](https://msty.app/)** — Modern, privacy-focused desktop assistant for local GGUF models with split-screen multi-model chats and prompt templates.
- **[Chatbox](https://chatboxai.app/)** — Clean cross-platform desktop client connecting to local llama.cpp / Ollama endpoints.
- **[Backyard AI](https://backyard.ai/)** (formerly Faraday) — Desktop application optimized for character-based conversational agents and local roleplay.
- **[AnythingLLM](https://github.com/Mintplex-Labs/anything-llm)** — Full-stack desktop app and enterprise solution with built-in RAG and local GGUF support.
- **[GPT4All](https://github.com/nomic-ai/gpt4all)** — Open-source ecosystem for running privacy-first LLMs locally on consumer CPUs and GPUs.

---

## ⚙️ Inference Engines & Serving

Production-grade engines and frameworks for hosting and serving GGUF models at scale:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — The gold-standard C/C++ inference engine created by Georgi Gerganov. Hardware acceleration across CUDA, Metal, ROCm, SYCL, Vulkan, and OpenCL.
- **[vLLM](https://github.com/vllm-project/vllm)** — High-throughput LLM serving engine with native support for loading GGUF format weights with PagedAttention.
- **[SGLang](https://github.com/sgl-project/sglang)** — Fast serving framework for complex prompting and multi-turn workflows with GGUF weight loading capabilities.
- **[Aphrodite Engine](https://github.com/PygmalionAI/aphrodite-engine)** — High-performance inference engine built on top of vLLM, specifically tailored for creative writing and local deployments.
- **[Kobold.cpp](https://github.com/LostRuins/koboldcpp)** — An all-in-one AI server bundling llama.cpp with KoboldAI's rich web interface.
- **[LocalAI](https://github.com/mudler/LocalAI)** — Drop-in replacement REST API compatible with OpenAI specifications for local and distributed inference.
- **[Tabby](https://github.com/TabbyML/tabby)** — Self-hosted AI coding assistant powering local code completions with GGUF backends.

---

## 📱 Mobile & Edge AI

Running GGUF models on mobile devices, Single Board Computers (SBCs), and edge hardware:

- **[PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai)** — Sleek, open-source native app to run GGUF models directly on iOS and Android devices.
- **[llama.cpp on iOS](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.swiftui)** — Official SwiftUI implementation showcasing on-device inference via Apple Silicon / Metal.
- **[MLC-LLM](https://github.com/mlc-ai/mlc-llm)** — Universal deployment engine bringing local language models to diverse hardware including Android, iPhone, and WebGPU.
- **[llama.cpp on Raspberry Pi](https://github.com/ggml-org/llama.cpp/discussions)** — Community guides and benchmarks for running 1B-3B quantized models on Raspberry Pi 4/5.

---

## 🛠️ Quantization & Model Tooling

Tools for converting, quantizing, inspecting, and editing GGUF files:

- **[llama.app](https://llama.app/)** — Interactive web-based GGUF metadata inspector, parameter analyzer, and memory calculator.
- **[convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)** — The primary conversion script from Hugging Face Safetensors/PyTorch to GGUF format in llama.cpp.
- **[ik_llama.cpp](https://github.com/Thireus/ik_llama.cpp)** — Advanced importance matrix (imatrix) quantization scripts and guides maintained by Thireus and Kalomaze.
- **[AutoGGUF](https://github.com/svjack/AutoGGUF)** — GUI tool for automating model conversion and quantization pipelines without writing bash scripts.
- **[gguf-parser](https://github.com/gpustack/gguf-parser)** — Fast Go/C/WASM library and CLI for inspecting GGUF file structures and estimating execution requirements.
- **[Unsloth](https://github.com/unslothai/unsloth)** — Ultra-fast fine-tuning library that provides direct 16-bit to GGUF export with integrated 4-bit / 8-bit quantization.

---

## 📚 SDKs & Language Bindings

Programmatic libraries to interact with GGUF files in your applications:

### Python
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** — Feature-complete Python bindings for llama.cpp with OpenAI-compatible client & web server.
- **[gguf-py](https://pypi.org/project/gguf/)** — Official Python package maintained by ggml-org for parsing and writing GGUF binary files.

### JavaScript & TypeScript
- **[@huggingface/gguf](https://github.com/huggingface/huggingface.js/tree/main/packages/gguf)** — Parse remote and local GGUF metadata and tensor headers in Node.js, Bun, or directly inside the browser.
- **[node-llama-cpp](https://github.com/withcatai/node-llama-cpp)** — Comprehensive Node.js native binding for llama.cpp with TypeScript support.
- **[Web-LLM](https://github.com/mlc-ai/web-llm)** — In-browser LLM inference engine accelerated by WebGPU.

### Rust
- **[candle](https://github.com/huggingface/candle)** — Minimalist ML framework for Rust with native support for reading and evaluating GGUF models.
- **[llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)** — Idiomatic Rust bindings for llama.cpp.

### Go
- **[go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)** — Go bindings for llama.cpp with CGo wrapper.

### .NET / C#
- **[LLamaSharp](https://github.com/SciSharp/LLamaSharp)** — Cross-platform C# / .NET bindings for llama.cpp.

### Swift
- **[llama.cpp Swift](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.swiftui)** — Official Swift package for Apple platforms.

---

## 🌐 Model Hubs & Top Quantizers

Where to discover, evaluate, and download pre-quantized GGUF models:

- **[Hugging Face GGUF Library Filter](https://huggingface.co/models?library=gguf)** — The primary directory indexing over 200,000+ ready-to-download GGUF models.
- **[bartowski](https://huggingface.co/bartowski)** — Top community quantizer providing rapid, high-quality k-quant and imatrix GGUF builds for newly released open-source models.
- **[MaziyarPanahi](https://huggingface.co/MaziyarPanahi)** — Prolific provider of quantized models spanning coding, reasoning, and multilingual domains.
- **[mradermacher](https://huggingface.co/mradermacher)** — Automated quantization pipeline contributor publishing diverse architecture quantizations.
- **[TheBloke](https://huggingface.co/TheBloke)** — Historical pioneer who standardized model releases in GGUF/GGML format across the early open-source boom.
- **[Unsloth](https://huggingface.co/unsloth)** — Official releases of fine-tuned and dynamic quantized checkpoints.

---

## 📖 Deep Dives, Specs & Articles

Essential reading material to understand the internal engineering of GGUF:

- **[GGUF Specification Document](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)** — The official ggml GGUF specification detailing header format, tensor metadata, and data alignment.
- **[Hugging Face GGUF Hub Documentation](https://huggingface.co/docs/hub/gguf)** — Guide on GGUF integration with the Hugging Face Hub, model viewer, and API.
- **[IBM: What is GGUF and GGML?](https://www.ibm.com/think/topics/gguf-versus-ggml)** — Architectural breakdown comparing GGUF against historical predecessor formats.
- **[Kalomaze's Guide on Importance Matrix](https://github.com/ggerganov/llama.cpp/pull/4861)** — The foundational proposal and mathematical explanation behind i-matrix quantization.

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Ensure the project or tool is actively maintained and directly related to the GGUF ecosystem.
2. Search existing entries to avoid duplicates.
3. Submit a Pull Request with a clear description and alphabetical order placement within its section.

Check [CONTRIBUTING.md](CONTRIBUTING.md) for full submission criteria.

---

## 📄 License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-0.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, the authors have waived all copyright and related rights to this work.

