# Awesome GGUF [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of awesome **GGUF** runtimes, tools, libraries, models, and comprehensive guides for running local Large Language Models (LLMs).

[English](README.md) | [Tiếng Việt 🇻🇳](README.vi.md)

---

## Contents

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
- [📄 License](#-license)

---

## 💡 What is GGUF?

**GGUF** is the binary format specification designed by [Georgi Gerganov](https://github.com/ggerganov) and the [ggml](https://github.com/ggml-org/ggml) team in August 2023. The name is often expanded as "GPT-Generated Unified Format", but this is an unofficial backronym; the specification itself does not define the acronym. It serves as the de-facto standard for storing and distributing quantized Large Language Models for local CPU and GPU inference.

### The Core Philosophy

Think of GGUF as the **`.mp3` or `.mkv` of AI models**:
- **Single-File Distribution:** Traditional models (like Hugging Face PyTorch/Safetensors checkpoints) are split across multiple weight shards, configuration JSONs, tokenizer definitions, and special token dictionaries. GGUF packs **the architecture, weights, tokenizer, and metadata into a single `.gguf` file**.
- **Extensible Key-Value Metadata:** Unlike legacy formats where modifying metadata broke parser compatibility, GGUF stores metadata in structured key-value pairs. New model architectures can be introduced without breaking backward compatibility for existing parsers.
- **Instant Loading via `mmap`:** GGUF is designed for direct memory-mapping (`mmap`). Operating systems can map the model file into RAM/VRAM instantly, bypassing slow sequential disk reads.
- **Native Quantization Support:** Built from the ground up to support reduced-precision integer representations (from 1.5-bit to 8-bit), allowing 70B+ parameter models to execute efficiently on consumer-grade hardware.

### GGUF vs. GGML vs. Safetensors

| Feature                    | Hugging Face Safetensors                      | GGML / GGJT (Legacy / Obsolete)                       | GGUF (Current Standard)                             |
|:---------------------------|:----------------------------------------------|:------------------------------------------------------|:----------------------------------------------------|
| **Primary Use-Case**       | Training, Fine-Tuning, Datacenter GPU         | Early llama.cpp inference (2023)                      | High-performance Local & Edge Inference             |
| **Packaging**              | Multi-file (`.json` configs + `.safetensors`) | Single file                                           | **Single self-contained binary file**               |
| **Metadata Flexibility**   | JSON header                                   | Hardcoded struct (brittle)                            | **Extensible Key-Value Store**                      |
| **Backward Compatibility** | High                                          | Poor (GGML → GGMF → GGJT v1/v2/v3 breaking changes)   | **High (Forward & Backward compatible)**            |
| **Quantization Variety**   | FP16, BF16, FP8, AWQ, GPTQ                    | Q4_0, Q4_1, Q5_x, Q8_0; K-Quants from GGJT v3         | **Legacy, K-Quants, I-Quants (IQ1–IQ4), imatrix**   |
| **Inference Hardware**     | High-VRAM GPUs (CUDA, ROCm)                   | CPU, early cuBLAS / CLBlast layer offload             | **CPU (AVX/NEON), Apple Metal, CUDA, ROCm, Vulkan** |

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
  - `_0` / `_1`: Legacy round-to-nearest block quantizations (`_1` adds a per-block minimum).
  - `_K`: **K-Quants** (introduced by ikawrakow in 2023). Block-wise quantization using 256-weight super-blocks with per-block scales and minimums. The `K` is just a name, not k-means clustering. `_S` / `_M` / `_L` variants mix bit widths across tensor types (e.g., attention vs. feed-forward).
  - `IQ`: **I-Quants** (importance-aware quants). Non-linear, codebook/lattice-based quantization for extreme low bitrates (1–4 bits). Usually built with an **importance matrix (imatrix)**, but the two are separate: an imatrix can also calibrate regular K-Quants (e.g., `Q4_K_M` with imatrix).
- **Scale (`_XXS`, `_XS`, `_S`, `_M`, `_L`):** Extra-Extra Small, Extra Small, Small, Medium, Large. Dictates which internal tensors are kept at higher bit precision.

### Quantization Types Overview

| Type                    | Quality Loss (Perplexity)* | Bits / Weight | Relative Size   | Recommendation / Best For                                      |
|:------------------------|:---------------------------|:--------------|:----------------|:---------------------------------------------------------------|
| **`Q8_0`**              | Negligible (~+0.01%)       | 8.5           | ~53% of FP16    | Near-lossless accuracy; baseline testing.                      |
| **`Q6_K`**              | Extremely Low (~+0.1%)     | 6.6           | ~41% of FP16    | Precision-critical tasks (coding, complex reasoning).          |
| **`Q5_K_M`**            | Very Low (~+0.2%)          | 5.7           | ~36% of FP16    | Balanced choice when RAM/VRAM is available.                    |
| **`Q4_K_M`**            | Low (~+0.9%)               | 4.8           | ~30% of FP16    | 🏆 **The Universal Sweet Spot** (best speed-to-quality ratio). |
| **`Q4_K_S`**            | Low-Moderate (~+1.7%)      | 4.6           | ~29% of FP16    | Slightly smaller footprint with minor quality degradation.     |
| **`IQ3_M` / `IQ3_S`**   | Moderate                   | 3.4–3.7       | ~22–23% of FP16 | Best choice for tight VRAM budget; outperforms legacy Q3.      |
| **`IQ2_M` / `IQ2_XXS`** | Noticeable                 | 2.1–2.7       | ~13–17% of FP16 | Extreme compression to fit 70B models on 16GB-24GB RAM.        |

*\* Perplexity deltas measured by llama.cpp on LLaMA 7B (wikitext). Larger models lose proportionally less at the same quant level.*

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

- **[AnythingLLM](https://github.com/Mintplex-Labs/anything-llm)** — Full-stack desktop app and enterprise solution with built-in RAG and local GGUF support.
- **[Backyard AI](https://backyard.ai/)** (formerly Faraday) — Desktop application optimized for character-based conversational agents and local roleplay. ⚠️ *Service shut down in 2026; user data export available until November 6, 2026.*
- **[Chatbox](https://chatboxai.app/)** — Clean cross-platform desktop client connecting to local llama.cpp / Ollama endpoints.
- **[GPT4All](https://github.com/nomic-ai/gpt4all)** — Open-source ecosystem for running privacy-first LLMs locally on consumer CPUs and GPUs. ⚠️ *Maintenance mode: last release v3.10.0 (February 2025), no commits since May 2025.*
- **[Jan.ai](https://github.com/janhq/jan)** — Open-source ChatGPT alternative running 100% locally on your machine with full GGUF support.
- **[Llama](https://llama.app/)** — Tiny menu bar / system tray app for macOS and Windows by the llama.cpp team and Hugging Face. Runs GGUF models with zero configuration and exposes them to editors, coding agents, and other apps.
- **[llamafile](https://github.com/mozilla-ai/llamafile)** — Combines `llama.cpp` and Cosmopolitan Libc into a single multi-gigabyte executable that runs locally on Windows, macOS, Linux, FreeBSD without installation.
- **[LM Studio](https://lmstudio.ai/)** — Desktop application to search Hugging Face GGUF models, download, chat, and run a local OpenAI-compatible API server.
- **[Msty](https://msty.ai/)** — Privacy-focused desktop assistant for local GGUF models with split-screen multi-model chats and prompt templates.
- **[Ollama](https://github.com/ollama/ollama)** — CLI tool, background service, and desktop app to get up and running with GGUF models via Modelfiles and REST API.
- **[Open WebUI](https://github.com/open-webui/open-webui)** — Self-hosted feature-packed AI interface with Ollama and OpenAI API compatibility, RAG, and multi-user management.

---

## ⚙️ Inference Engines & Serving

Production-grade engines and frameworks for hosting and serving GGUF models at scale:

- **[KoboldCpp](https://github.com/LostRuins/koboldcpp)** — An all-in-one AI server bundling llama.cpp with KoboldAI's rich web interface.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — The reference C/C++ inference engine created by Georgi Gerganov. Hardware acceleration across CUDA, Metal, ROCm (HIP), SYCL, Vulkan, and OpenCL.
- **[LocalAI](https://github.com/mudler/LocalAI)** — Drop-in replacement REST API compatible with OpenAI specifications for local and distributed inference.
- **[SGLang](https://github.com/sgl-project/sglang)** — Fast serving framework for complex prompting and multi-turn workflows with a dedicated GGUF model loader.
- **[Sonar](https://github.com/dphnAI/sonar)** (formerly Aphrodite Engine) — High-performance inference engine built on top of vLLM with CUDA GGUF kernels, tailored for creative writing and large-scale deployments.
- **[Tabby](https://github.com/TabbyML/tabby)** — Self-hosted AI coding assistant powering local code completions with GGUF backends.
- **[vLLM](https://github.com/vllm-project/vllm)** — High-throughput LLM serving engine with experimental support for loading GGUF format weights with PagedAttention.

---

## 📱 Mobile & Edge AI

Running GGUF models on mobile devices, Single Board Computers (SBCs), and edge hardware:

- **[llama.cpp on Arm CPUs / Raspberry Pi](https://github.com/ggml-org/llama.cpp/discussions/30069)** — Official community benchmark thread (`llama-bench`) for Arm CPUs including Raspberry Pi 5, RK3588, Snapdragon, Graviton, and Ampere Altra.
- **[llama.cpp on iOS](https://github.com/ggml-org/llama.cpp/tree/master/examples/llama.swiftui)** — Official SwiftUI example showcasing on-device inference via Apple Silicon / Metal.
- **[MLC-LLM](https://github.com/mlc-ai/mlc-llm)** — Universal deployment engine bringing local language models to Android, iOS, and WebGPU. ⚠️ *Does not read GGUF; uses its own TVM-compiled MLC weight format. Listed as a related alternative.*
- **[PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai)** — Open-source native app to run GGUF models directly on iOS and Android devices.

---

## 🛠️ Quantization & Model Tooling

Tools for converting, quantizing, inspecting, and editing GGUF files:

- **[AutoGGUF](https://github.com/leafspark/AutoGGUF)** — GUI tool for automating model conversion, imatrix generation, and quantization pipelines without writing shell scripts. ⚠️ *Repository archived in December 2025; last release v2.0.1 (May 2025). Still functional.*
- **[convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)** — The primary conversion script from Hugging Face Safetensors/PyTorch to GGUF format in llama.cpp.
- **[gguf-parser-go](https://github.com/gpustack/gguf-parser-go)** — Go library and CLI that reads GGUF metadata from local or remote files (without downloading the whole file) and estimates memory usage and tokens per second.
- **[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** — llama.cpp fork by ikawrakow (author of K-Quants and I-Quants) with additional state-of-the-art quantization types and CPU/GPU performance improvements.
- **[Unsloth](https://github.com/unslothai/unsloth)** — Ultra-fast fine-tuning library that provides direct 16-bit to GGUF export with integrated 4-bit / 8-bit quantization.

---

## 📚 SDKs & Language Bindings

Programmatic libraries to interact with GGUF files in your applications:

### Python
- **[gguf-py](https://pypi.org/project/gguf/)** — Official Python package maintained by ggml-org for parsing and writing GGUF binary files.
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** — Feature-complete Python bindings for llama.cpp with OpenAI-compatible client & web server.

### JavaScript & TypeScript
- **[@huggingface/gguf](https://github.com/huggingface/huggingface.js/tree/main/packages/gguf)** — Parse remote and local GGUF metadata and tensor headers in Node.js, Bun, or directly inside the browser.
- **[node-llama-cpp](https://github.com/withcatai/node-llama-cpp)** — Comprehensive Node.js native binding for llama.cpp with TypeScript support.
- **[Web-LLM](https://github.com/mlc-ai/web-llm)** — In-browser LLM inference engine accelerated by WebGPU. ⚠️ *Does not read GGUF; uses the MLC compiled format. Listed as a related alternative.*

### Rust
- **[candle](https://github.com/huggingface/candle)** — Minimalist ML framework for Rust with native support for reading and evaluating GGUF models.
- **[llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)** — Idiomatic Rust bindings for llama.cpp.

### Go
- **[go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)** — Go bindings for llama.cpp with CGo wrapper.

### .NET / C#
- **[LLamaSharp](https://github.com/SciSharp/LLamaSharp)** — Cross-platform C# / .NET bindings for llama.cpp.

### Swift
- **[llama.cpp XCFramework](https://github.com/ggml-org/llama.cpp/blob/master/build-xcframework.sh)** — Official script that builds llama.cpp as an XCFramework for iOS, macOS, visionOS, and tvOS (replaces the removed root `Package.swift`).

---

## 🌐 Model Hubs & Top Quantizers

Where to discover, evaluate, and download pre-quantized GGUF models:

- **[Hugging Face GGUF Library Filter](https://huggingface.co/models?library=gguf)** — The primary directory indexing over 200,000+ ready-to-download GGUF models.
- **[bartowski](https://huggingface.co/bartowski)** — Top community quantizer providing rapid, high-quality k-quant and imatrix GGUF builds for newly released open-source models.
- **[DevQuasar](https://huggingface.co/DevQuasar)** — High-volume quantizer covering a broad range of community fine-tunes and niche architectures.
- **[ggml-org](https://huggingface.co/ggml-org)** — Official Hugging Face organization of the ggml / llama.cpp team. Reference GGUF conversions of popular models, used in llama.cpp docs and examples.
- **[lmstudio-community](https://huggingface.co/lmstudio-community)** — LM Studio team's curated day-one GGUF builds of major releases, verified to run in LM Studio.
- **[MaziyarPanahi](https://huggingface.co/MaziyarPanahi)** — Prolific provider of quantized models spanning coding, reasoning, and multilingual domains.
- **[mradermacher](https://huggingface.co/mradermacher)** — Automated quantization pipeline contributor publishing diverse architecture quantizations.
- **[Mungert](https://huggingface.co/Mungert)** — imatrix GGUF builds with mixed-precision ultra-low-bit variants aimed at memory-constrained and edge devices.
- **[TheBloke](https://huggingface.co/TheBloke)** — Historical pioneer who standardized model releases in GGUF/GGML format across the early open-source boom (inactive since early 2024).
- **[ubergarm](https://huggingface.co/ubergarm)** — Specialized quants of large MoE models (DeepSeek, Qwen3, Kimi) using ik_llama.cpp-exclusive quant types; not loadable by mainline llama.cpp.
- **[Unsloth](https://huggingface.co/unsloth)** — Official releases of fine-tuned and dynamic quantized checkpoints.

---

## 📖 Deep Dives, Specs & Articles

Essential reading material to understand the internal engineering of GGUF:

- **[GGUF Specification Document](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)** — The official ggml GGUF specification detailing header format, tensor metadata, and data alignment.
- **[Hugging Face GGUF Hub Documentation](https://huggingface.co/docs/hub/gguf)** — Guide on GGUF integration with the Hugging Face Hub, model viewer, and API.
- **[IBM: GGUF versus GGML](https://www.ibm.com/think/topics/gguf-versus-ggml)** — Architectural breakdown comparing GGUF against historical predecessor formats.
- **[Importance Matrix calculation (llama.cpp PR #4861)](https://github.com/ggml-org/llama.cpp/pull/4861)** — ikawrakow's foundational pull request introducing imatrix computation and the reasoning behind importance-weighted quantization.

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Ensure the project or tool is actively maintained and directly related to the GGUF ecosystem.
2. Search existing entries to avoid duplicates.
3. Submit a Pull Request with a clear description and alphabetical order placement within its section.

Check [CONTRIBUTING.md](CONTRIBUTING.md) for full submission criteria.

---

## 📄 License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, the authors have waived all copyright and related rights to this work. See [LICENSE](LICENSE).
