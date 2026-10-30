# Awesome GGUF [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> Tuyển tập toàn diện về định dạng **GGUF**: các runtime, công cụ, thư viện, mô hình và cẩm nang hướng dẫn chuyên sâu để chạy Large Language Model (LLM) cục bộ.

[English](README.md) | [Tiếng Việt 🇻🇳](README.vi.md)

---

## Mục lục

- [💡 GGUF là gì?](#-gguf-là-gì)
  - [Triết lý cốt lõi](#triết-lý-cốt-lõi)
  - [So sánh: GGUF vs. GGML vs. Safetensors](#so-sánh-gguf-vs-ggml-vs-safetensors)
- [🎛️ Cẩm nang Quantization (Lượng tử hóa)](#️-cẩm-nang-quantization-lượng-tử-hóa)
  - [Giải mã tên file GGUF](#giải-mã-tên-file-gguf)
  - [Bảng tổng hợp các chuẩn Quant](#bảng-tổng-hợp-các-chuẩn-quant)
  - [Công thức tính RAM / VRAM phần cứng](#công-thức-tính-ram--vram-phần-cứng)
- [🖥️ Ứng dụng Desktop (Cài là chạy)](#️-ứng-dụng-desktop-cài-là-chạy)
- [⚙️ Inference Engines & Serving](#️-inference-engines--serving)
- [📱 Mobile & Edge AI](#-mobile--edge-ai)
- [🛠️ Công cụ Quantization & Quản lý Model](#️-công-cụ-quantization--quản-lý-model)
- [📚 SDKs & Thư viện lập trình](#-sdks--thư-viện-lập-trình)
- [🌐 Kho tải Model & Top Quantizers](#-kho-tải-model--top-quantizers)
- [📖 Tài liệu chuyên sâu & Thông số kỹ thuật](#-tài-liệu-chuyên-sâu--thông-số-kỹ-thuật)
- [🤝 Đóng góp (Contributing)](#-đóng-góp-contributing)
- [📄 Bản quyền (License)](#-bản-quyền-license)

---

## 💡 GGUF là gì?

**GGUF** là đặc tả định dạng nhị phân được [Georgi Gerganov](https://github.com/ggerganov) và đội ngũ phát triển [ggml](https://github.com/ggml-org/ggml) giới thiệu vào tháng 8 năm 2023. Tên này thường được diễn giải là "GPT-Generated Unified Format", nhưng đó là cách giải nghĩa không chính thức; bản đặc tả gốc không định nghĩa từ viết tắt này. GGUF hiện là tiêu chuẩn de-facto để lưu trữ và phân phối các mô hình ngôn ngữ lớn (LLM) đã qua lượng tử hóa, tối ưu cho việc suy luận (inference) trên CPU và GPU máy tính cá nhân.

### Triết lý cốt lõi

Hãy hình dung GGUF giống như định dạng **`.mp3` hoặc `.mkv` trong thế giới AI**:
- **Đóng gói tất cả trong một (Single-file Distribution):** Các mô hình truyền thống (như PyTorch/Safetensors trên Hugging Face) thường bị phân mảnh thành nhiều mảnh trọng số (weight shards), các file cấu hình `.json`, từ điển tokenizer riêng lẻ. GGUF gom toàn bộ: **kiến trúc mô hình, trọng số, tokenizer và siêu dữ liệu (metadata) vào duy nhất 1 file `.gguf`**.
- **Siêu dữ liệu Key-Value linh hoạt:** Khắc phục nhược điểm chí mạng của định dạng GGML cũ (mỗi lần cập nhật mô hình mới là làm vỡ tính tương thích của parser). GGUF lưu metadata dưới dạng cặp Khóa - Giá trị (Key-Value), cho phép thêm các kiến trúc mô hình mới mà không làm hỏng các công cụ đọc hiện có.
- **Tải tức thì bằng `mmap` (Memory Mapping):** GGUF được thiết kế tối ưu cho kỹ thuật `mmap`. Hệ điều hành có thể ánh xạ trực tiếp file từ ổ cứng vào RAM/VRAM gần như ngay lập tức, bỏ qua quá trình đọc tuần tự từng byte chậm chạp.
- **Tối ưu hóa sâu cho Quantization:** Hỗ trợ bản địa các mức lượng tử hóa từ 1.5-bit đến 8-bit, cho phép những mô hình khổng lồ như 70B tham số có thể chạy mượt mà trên phần cứng người dùng phổ thông.

### So sánh: GGUF vs. GGML vs. Safetensors

| Tiêu chí                  | Hugging Face Safetensors                          | GGML / GGJT (Cũ / Đã ngừng phát triển)              | GGUF (Tiêu chuẩn hiện nay)                          |
|:--------------------------|:--------------------------------------------------|:----------------------------------------------------|:----------------------------------------------------|
| **Mục đích chính**        | Training, Fine-tuning, GPU Data Center            | Inference llama.cpp thời kỳ đầu (2023)              | **Inference tối ưu trên Local & Edge**              |
| **Quy cách đóng gói**     | Nhiều file (file config `.json` + `.safetensors`) | 1 file                                              | **1 file nhị phân độc lập duy nhất**                |
| **Độ linh hoạt Metadata** | Header JSON                                       | Struct cố định (rất dễ gãy format)                  | **Cặp Key-Value mở rộng linh hoạt**                 |
| **Tương thích ngược**     | Tốt                                               | Kém (GGML → GGMF → GGJT v1/v2/v3 đều gãy format)    | **Xuất sắc (Tương thích xuôi & ngược)**             |
| **Các kiểu Quantization** | FP16, BF16, FP8, AWQ, GPTQ                        | Q4_0, Q4_1, Q5_x, Q8_0; K-Quants từ GGJT v3         | **Legacy, K-Quants, I-Quants (IQ1–IQ4), imatrix**   |
| **Hỗ trợ phần cứng**      | GPU chuyên dụng (CUDA, ROCm)                      | CPU, offload layer sớm qua cuBLAS / CLBlast         | **CPU (AVX/NEON), Apple Metal, CUDA, ROCm, Vulkan** |

---

## 🎛️ Cẩm nang Quantization (Lượng tử hóa)

### Giải mã tên file GGUF

Một tên file GGUF tiêu chuẩn phản ánh đầy đủ thông tin mô hình, phiên bản và phương pháp nén:

```
Llama-3.1-8B-Instruct.Q4_K_M.gguf
│─────────┬─────────│ │───┬───│
          │               └── Phương pháp nén (4-bit, K-Quant, kích cỡ Medium)
          └────────────────── Tên kiến trúc, số tham số và phiên bản fine-tune
```

- **Mức Bit (`Q4`, `Q5`, `Q8`, `IQ3`):** Độ chính xác mục tiêu (ví dụ: `4` bit cho mỗi trọng số thay vì 16-bit ở bản gốc FP16).
- **Phân loại phương pháp:**
  - `_0` / `_1`: Các thuật toán làm tròn theo block đời đầu (`_1` thêm giá trị min cho mỗi block).
  - `_K`: **K-Quants** (do ikawrakow giới thiệu năm 2023). Lượng tử hóa theo block, dùng super-block 256 trọng số với scale và min riêng cho từng block. Chữ `K` chỉ là tên gọi, **không phải** phân cụm k-means. Các biến thể `_S` / `_M` / `_L` phân bổ số bit khác nhau giữa các loại tensor (ví dụ attention và feed-forward).
  - `IQ`: **I-Quants** (quant nhận biết mức độ quan trọng). Lượng tử hóa phi tuyến dựa trên codebook/lattice để nén cực sâu (1–4 bit). Thường được tạo kèm **importance matrix (imatrix)**, nhưng hai khái niệm này tách biệt: imatrix cũng có thể dùng để hiệu chỉnh K-Quants thông thường (ví dụ `Q4_K_M` kèm imatrix).
- **Quy mô kích thước (`_XXS`, `_XS`, `_S`, `_M`, `_L`):** Extra-Extra Small, Extra Small, Small, Medium, Large (chỉ định các tensor quan trọng bên trong được giữ ở mức bit cao hơn).

### Bảng tổng hợp các chuẩn Quant

| Loại Quant              | Mức suy giảm chất lượng (Perplexity)* | Bit / trọng số | Dung lượng so với FP16 | Lời khuyên & Mục đích sử dụng                                                           |
|:------------------------|:--------------------------------------|:---------------|:-----------------------|:----------------------------------------------------------------------------------------|
| **`Q8_0`**              | Gần như bằng 0 (~+0.01%)              | 8.5            | ~53% dung lượng gốc    | Giữ nguyên độ chính xác; dùng làm chuẩn đối sánh.                                       |
| **`Q6_K`**              | Cực kỳ thấp (~+0.1%)                  | 6.6            | ~41% dung lượng gốc    | Tác vụ đòi hỏi độ chính xác cao (viết code, toán học).                                  |
| **`Q5_K_M`**            | Rất thấp (~+0.2%)                     | 5.7            | ~36% dung lượng gốc    | Lựa chọn tuyệt vời khi dung lượng RAM/VRAM cho phép.                                    |
| **`Q4_K_M`**            | Thấp (~+0.9%)                         | 4.8            | ~30% dung lượng gốc    | 🏆 **Điểm ngọt quốc dân (Sweet Spot)**: Cân bằng hoàn hảo giữa tốc độ và độ thông minh. |
| **`Q4_K_S`**            | Thấp - Vừa (~+1.7%)                   | 4.6            | ~29% dung lượng gốc    | Nhẹ hơn bản M một chút, đánh đổi một phần nhỏ chất lượng.                               |
| **`IQ3_M` / `IQ3_S`**   | Trung bình                            | 3.4–3.7        | ~22–23% dung lượng gốc | Khuyên dùng khi VRAM eo hẹp; vượt trội hơn hẳn Q3 cũ.                                   |
| **`IQ2_M` / `IQ2_XXS`** | Rõ rệt                                | 2.1–2.7        | ~13–17% dung lượng gốc | Nén tối đa để cố gắng nhét mô hình 70B vào RAM 16GB-24GB.                               |

*\* Số liệu perplexity do llama.cpp đo trên LLaMA 7B (wikitext). Mô hình càng lớn thì suy giảm càng ít ở cùng mức quant.*

### Công thức tính RAM / VRAM phần cứng

Ước tính bộ nhớ cần thiết để chạy một mô hình GGUF:

$$\text{Dung lượng RAM/VRAM (GB)} \approx \left(\text{Số tỷ tham số (B)} \times \frac{\text{Số bit}}{8} \times 1.2\right) + \text{Bộ nhớ KV Cache (1-2 GB)}$$

*Quy chuẩn tham khảo nhanh:*
- **8 GB Unified Memory / VRAM:** Chạy tốt mô hình **7B / 8B** ở mức `Q4_K_M`.
- **16 GB Unified Memory / VRAM:** Chạy mô hình **7B / 8B** ở mức `Q8_0` hoặc **14B** ở mức `Q4_K_M`.
- **24 GB VRAM (RTX 3090 / 4090):** Chạy mượt mô hình **32B** ở mức `Q4_K_M` hoặc **70B** ở mức `IQ2_XS`.
- **64 GB+ Unified Memory (Mac Studio/Pro):** Chạy thoải mái mô hình **70B** ở mức `Q4_K_M` / `Q5_K_M` với context dài.

---

## 🖥️ Ứng dụng Desktop (Cài là chạy)

Các phần mềm đồ họa hoàn chỉnh, trực quan, không yêu cầu kỹ năng lập trình:

- **[AnythingLLM](https://github.com/Mintplex-Labs/anything-llm)** — Ứng dụng tích hợp RAG mạnh mẽ, quản lý tài liệu doanh nghiệp và chạy GGUF cục bộ.
- **[Backyard AI](https://backyard.ai/)** (tên cũ Faraday) — Tối ưu hóa chuyên sâu cho các nhân vật AI hội thoại và nhập vai (roleplay) ngoại tuyến. ⚠️ *Dịch vụ đã đóng năm 2026; người dùng có thể xuất dữ liệu đến 06/11/2026.*
- **[Chatbox](https://chatboxai.app/)** — Client đa nền tảng kết nối trực tiếp đến các backend llama.cpp / Ollama nội bộ.
- **[GPT4All](https://github.com/nomic-ai/gpt4all)** — Hệ sinh thái mã nguồn mở tập trung vào quyền riêng tư trên CPU và GPU phổ thông. ⚠️ *Chế độ bảo trì: bản phát hành cuối v3.10.0 (02/2025), không có commit từ 05/2025.*
- **[Jan.ai](https://github.com/janhq/jan)** — Giải pháp thay thế ChatGPT mã nguồn mở chạy offline 100% trên máy tính với giao diện tinh tế.
- **[Llama](https://llama.app/)** — Ứng dụng menu bar / system tray siêu nhẹ cho macOS và Windows do đội llama.cpp và Hugging Face phát triển. Chạy model GGUF không cần cấu hình, kết nối được với editor, coding agent và các app khác.
- **[llamafile](https://github.com/mozilla-ai/llamafile)** — Dự án kết hợp `llama.cpp` và Cosmopolitan Libc thành một file nhị phân duy nhất chạy được trên Windows, macOS, Linux mà không cần cài đặt bất kỳ dependency nào.
- **[LM Studio](https://lmstudio.ai/)** — Ứng dụng desktop để tìm kiếm mô hình GGUF trên Hugging Face, tải về bằng một cú click, chat và mở local OpenAI API server.
- **[Msty](https://msty.ai/)** — Ứng dụng desktop bảo mật, hỗ trợ so sánh song song nhiều mô hình và quản lý thư viện prompt.
- **[Ollama](https://github.com/ollama/ollama)** — Công cụ CLI, background daemon và desktop app để quản lý và chạy mô hình GGUF thông qua Modelfile và REST API.
- **[Open WebUI](https://github.com/open-webui/open-webui)** — Giao diện web UI tự host giàu tính năng, tích hợp sâu với Ollama, hỗ trợ RAG và quản lý nhiều người dùng.

---

## ⚙️ Inference Engines & Serving

Các engine hiệu năng cao dùng để triển khai và phục vụ (serving) mô hình ở quy mô lớn:

- **[KoboldCpp](https://github.com/LostRuins/koboldcpp)** — Server AI all-in-one tích hợp llama.cpp cùng giao diện web phong phú của KoboldAI.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — Cội nguồn của toàn bộ hệ sinh thái GGUF do Georgi Gerganov khởi xướng. Viết bằng C/C++, tăng tốc trên CUDA, Metal (Apple Silicon), ROCm (HIP), SYCL, Vulkan và OpenCL.
- **[LocalAI](https://github.com/mudler/LocalAI)** — REST API thay thế tương thích chuẩn OpenAI để phục vụ suy luận cục bộ và phân tán.
- **[SGLang](https://github.com/sgl-project/sglang)** — Framework serving tốc độ cao cho các luồng xử lý prompt phức tạp, có model loader riêng cho GGUF.
- **[Sonar](https://github.com/dphnAI/sonar)** (tên cũ Aphrodite Engine) — Engine inference hiệu năng cao dựa trên vLLM, có kernel CUDA cho GGUF, chuyên dụng cho sáng tạo nội dung và triển khai quy mô lớn.
- **[Tabby](https://github.com/TabbyML/tabby)** — Trợ lý viết code tự host, hoạt động tương tự GitHub Copilot với backend GGUF.
- **[vLLM](https://github.com/vllm-project/vllm)** — Engine serving thông lượng cao, hỗ trợ (thử nghiệm) nạp trực tiếp trọng số GGUF kết hợp cùng thuật toán PagedAttention.

---

## 📱 Mobile & Edge AI

Triển khai mô hình GGUF trên điện thoại di động, máy tính bảng và chip nhúng:

- **[llama.cpp trên Arm CPU / Raspberry Pi](https://github.com/ggml-org/llama.cpp/discussions/30069)** — Thread benchmark cộng đồng chính thức (`llama-bench`) cho Arm CPU gồm Raspberry Pi 5, RK3588, Snapdragon, Graviton và Ampere Altra.
- **[llama.cpp on iOS](https://github.com/ggml-org/llama.cpp/tree/master/examples/llama.swiftui)** — Ví dụ mẫu SwiftUI chính thức khai thác Metal trên chip Apple A-series/M-series.
- **[MLC-LLM](https://github.com/mlc-ai/mlc-llm)** — Nền tảng biên dịch và tăng tốc đa nền tảng cho Android, iOS và WebGPU. ⚠️ *Không đọc GGUF; dùng định dạng MLC riêng biên dịch qua TVM. Liệt kê như một giải pháp thay thế liên quan.*
- **[PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai)** — Ứng dụng mã nguồn mở chạy trực tiếp các mô hình GGUF trên iPhone, iPad và điện thoại Android.

---

## 🛠️ Công cụ Quantization & Quản lý Model

Các tiện ích chuyển đổi, phân tích metadata và nén trọng số:

- **[AutoGGUF](https://github.com/leafspark/AutoGGUF)** — Giao diện đồ họa tự động hóa quy trình chuyển đổi, tạo imatrix và quantize model mà không cần viết script. ⚠️ *Repo đã archived 12/2025; bản phát hành cuối v2.0.1 (05/2025). Vẫn dùng được.*
- **[convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)** — Script chính thức chuyển đổi từ Hugging Face PyTorch/Safetensors sang định dạng GGUF.
- **[gguf-parser-go](https://github.com/gpustack/gguf-parser-go)** — Thư viện Go và CLI đọc metadata GGUF từ file local hoặc remote (không cần tải toàn bộ file), ước tính RAM cần dùng và tốc độ token/giây.
- **[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** — Fork llama.cpp của ikawrakow (tác giả K-Quants và I-Quants) bổ sung thêm các kiểu quant tiên tiến và cải thiện hiệu năng CPU/GPU.
- **[Unsloth](https://github.com/unslothai/unsloth)** — Thư viện fine-tune siêu tốc hỗ trợ xuất thẳng model sau huấn luyện sang GGUF 4-bit / 8-bit.

---

## 📚 SDKs & Thư viện lập trình

Thư viện tích hợp GGUF vào các dự án phần mềm theo từng ngôn ngữ:

### Python
- **[gguf-py](https://pypi.org/project/gguf/)** — Package Python chính thức từ ggml-org để đọc và ghi cấu trúc nhị phân GGUF.
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** — Bindings Python hoàn chỉnh cho llama.cpp, tích hợp sẵn web server tương thích chuẩn OpenAI.

### JavaScript & TypeScript
- **[@huggingface/gguf](https://github.com/huggingface/huggingface.js/tree/main/packages/gguf)** — Đọc siêu dữ liệu và tensor header của file GGUF từ xa (remote) hoặc nội bộ mà không cần tải toàn bộ file về máy.
- **[node-llama-cpp](https://github.com/withcatai/node-llama-cpp)** — Thư viện native C++ binding mạnh mẽ cho Node.js và Bun hỗ trợ TypeScript hoàn chỉnh.
- **[Web-LLM](https://github.com/mlc-ai/web-llm)** — Chạy suy luận trực tiếp trong trình duyệt web bằng WebGPU. ⚠️ *Không đọc GGUF; dùng định dạng biên dịch MLC. Liệt kê như một giải pháp thay thế liên quan.*

### Rust
- **[candle](https://github.com/huggingface/candle)** — Framework Machine Learning tinh gọn của Hugging Face viết bằng Rust, hỗ trợ đọc và chạy mô hình GGUF.
- **[llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)** — Rust bindings an toàn cho llama.cpp.

### Go
- **[go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)** — Go bindings tương tác với llama.cpp thông qua CGo.

### .NET / C#
- **[LLamaSharp](https://github.com/SciSharp/LLamaSharp)** — C# / .NET bindings đa nền tảng để chạy llama.cpp trên Windows, Linux và macOS.

### Swift
- **[llama.cpp XCFramework](https://github.com/ggml-org/llama.cpp/blob/master/build-xcframework.sh)** — Script chính thức build llama.cpp thành XCFramework cho iOS, macOS, visionOS và tvOS (thay thế file `Package.swift` đã bị gỡ bỏ).

---

## 🌐 Kho tải Model & Top Quantizers

Địa chỉ tìm kiếm và tải về các mô hình GGUF chuẩn chất lượng cao:

- **[Bộ lọc thư viện GGUF trên Hugging Face](https://huggingface.co/models?library=gguf)** — Danh mục chính thức với hơn 200.000+ mô hình GGUF sẵn sàng tải về.
- **[bartowski](https://huggingface.co/bartowski)** — Quantizer cộng đồng uy tín, liên tục cập nhật các bản quantize k-quant và imatrix chất lượng cao ngay khi có model mới.
- **[DevQuasar](https://huggingface.co/DevQuasar)** — Quantizer sản lượng lớn, phủ rộng các bản fine-tune cộng đồng và kiến trúc ít phổ biến.
- **[ggml-org](https://huggingface.co/ggml-org)** — Tổ chức Hugging Face chính thức của đội ggml / llama.cpp. Các bản GGUF tham chiếu của model phổ biến, dùng trong docs và ví dụ của llama.cpp.
- **[lmstudio-community](https://huggingface.co/lmstudio-community)** — Bản GGUF do đội LM Studio tuyển chọn, phát hành ngay ngày đầu cho các model lớn, đã kiểm chứng chạy trong LM Studio.
- **[MaziyarPanahi](https://huggingface.co/MaziyarPanahi)** — Đóng góp số lượng lớn mô hình chuyên biệt về coding, suy luận logic và đa ngôn ngữ.
- **[mradermacher](https://huggingface.co/mradermacher)** — Vận hành hệ thống quantize tự động quy mô lớn với nhiều kiến trúc đa dạng.
- **[Mungert](https://huggingface.co/Mungert)** — Bản GGUF imatrix với các biến thể ultra-low-bit trộn độ chính xác, nhắm tới thiết bị edge và máy ít bộ nhớ.
- **[TheBloke](https://huggingface.co/TheBloke)** — Người tiên phong định hình làn sóng phân phối GGML/GGUF trong giai đoạn đầu của mã nguồn mở (ngừng hoạt động từ đầu 2024).
- **[ubergarm](https://huggingface.co/ubergarm)** — Quant chuyên biệt cho các model MoE lớn (DeepSeek, Qwen3, Kimi) dùng kiểu quant riêng của ik_llama.cpp; llama.cpp gốc không load được.
- **[Unsloth](https://huggingface.co/unsloth)** — Kênh phát hành chính thức các mô hình fine-tune và dynamic quant.

---

## 📖 Tài liệu chuyên sâu & Thông số kỹ thuật

Tài liệu kỹ thuật nền tảng để nghiên cứu sâu về kiến trúc GGUF:

- **[Tài liệu đặc tả kỹ thuật GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)** — Tài liệu chính thức quy định cấu trúc header, metadata key-value và căn chỉnh byte (data alignment).
- **[Tài liệu Hugging Face GGUF Hub](https://huggingface.co/docs/hub/gguf)** — Hướng dẫn chi tiết về cách Hugging Face tích hợp parser và viewer cho file GGUF.
- **[IBM: GGUF versus GGML](https://www.ibm.com/think/topics/gguf-versus-ggml)** — Bài phân tích kiến trúc so sánh sự khác biệt giữa hai thế hệ định dạng.
- **[Importance Matrix calculation (llama.cpp PR #4861)](https://github.com/ggml-org/llama.cpp/pull/4861)** — Pull request nền tảng của ikawrakow giới thiệu cách tính imatrix và cơ sở lý luận của lượng tử hóa theo trọng số quan trọng.

---

## 🤝 Đóng góp (Contributing)

Mọi đóng góp từ cộng đồng đều được hoan nghênh! Hãy tuân theo các nguyên tắc sau:
1. Đảm bảo công cụ hoặc dự án vẫn đang được duy trì tích cực và liên quan trực tiếp đến hệ sinh thái GGUF.
2. Kiểm tra danh sách hiện có để tránh trùng lặp.
3. Gửi Pull Request với mô tả rõ ràng và sắp xếp theo thứ tự bảng chữ cái trong danh mục tương ứng.

Xem chi tiết hướng dẫn tại [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 Bản quyền (License)

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

Theo mức độ tối đa được luật pháp cho phép, các tác giả từ bỏ mọi quyền tác giả và quyền liên quan đối với công trình này. Xem [LICENSE](LICENSE).
