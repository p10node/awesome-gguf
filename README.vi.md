# Awesome GGUF [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> Tuyển tập toàn diện về định dạng **GGUF**: các runtime, công cụ, thư viện, mô hình và cẩm nang hướng dẫn chuyên sâu để chạy Large Language Model (LLM) cục bộ.

[English](README.md) | [Tiếng Việt 🇻🇳](README.vi.md)

---

## 📑 Mục lục

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

---

## 💡 GGUF là gì?

**GGUF** (viết tắt của *GPT-Generated Unified Format*) là đặc tả định dạng nhị phân được [Georgi Gerganov](https://github.com/ggerganov) và đội ngũ phát triển [ggml](https://github.com/ggml-org/ggml) giới thiệu vào tháng 8 năm 2023. GGUF hiện là tiêu chuẩn vàng (de-facto standard) để lưu trữ và phân phối các mô hình ngôn ngữ lớn (LLM) đã qua lượng tử hóa, tối ưu cho việc suy luận (inference) trên CPU và GPU máy tính cá nhân.

### Triết lý cốt lõi

Hãy hình dung GGUF giống như định dạng **`.mp3` hoặc `.mkv` trong thế giới AI**:
- **Đóng gói tất cả trong một (Single-file Distribution):** Các mô hình truyền thống (như PyTorch/Safetensors trên Hugging Face) thường bị phân mảnh thành nhiều mảnh trọng số (weight shards), các file cấu hình `.json`, từ điển tokenizer riêng lẻ. GGUF gom toàn bộ: **kiến trúc mô hình, trọng số, tokenizer và siêu dữ liệu (metadata) vào duy nhất 1 file `.gguf`**.
- **Siêu dữ liệu Key-Value linh hoạt:** Khắc phục nhược điểm chí mạng của định dạng GGML cũ (mỗi lần cập nhật mô hình mới là làm vỡ tính tương thích của parser). GGUF lưu metadata dưới dạng cặp Khóa - Giá trị (Key-Value), cho phép thêm các kiến trúc mô hình mới mà không làm hỏng các công cụ đọc hiện có.
- **Tải tức thì bằng `mmap` (Memory Mapping):** GGUF được thiết kế tối ưu cho kỹ thuật `mmap`. Hệ điều hành có thể ánh xạ trực tiếp file từ ổ cứng vào RAM/VRAM gần như ngay lập tức, bỏ qua quá trình đọc tuần tự từng byte chậm chạp.
- **Tối ưu hóa sâu cho Quantization:** Hỗ trợ bản địa các mức lượng tử hóa từ 1.5-bit đến 8-bit, cho phép những mô hình khổng lồ như 70B tham số có thể chạy mượt mà trên phần cứng người dùng phổ thông.

### So sánh: GGUF vs. GGML vs. Safetensors

| Tiêu chí                  | Hugging Face Safetensors                          | GGML (Cũ / Đã ngừng phát triển)         | GGUF (Tiêu chuẩn hiện nay)                    |
|:--------------------------|:--------------------------------------------------|:----------------------------------------|:----------------------------------------------|
| **Mục đích chính**        | Training, Fine-tuning, GPU Data Center            | Chạy inference CPU đời đầu              | **Inference tối ưu trên Local & Edge**        |
| **Quy cách đóng gói**     | Nhiều file (file config `.json` + `.safetensors`) | 1 file                                  | **1 file nhị phân độc lập duy nhất**          |
| **Độ linh hoạt Metadata** | Header JSON                                       | Struct cố định (rất dễ gãy format)      | **Cặp Key-Value mở rộng linh hoạt**           |
| **Tương thích ngược**     | Tốt                                               | Kém (mỗi phiên bản lại phải đổi format) | **Xuất sắc (Tương thích xuôi & ngược)**       |
| **Các kiểu Quantization** | FP16, BF16, FP8, AWQ, GPTQ                        | Cơ bản (Q4_0, Q4_1, Q8_0)               | **K-Quants, I-Matrix (IQ1 đến IQ8)**          |
| **Hỗ trợ phần cứng**      | GPU chuyên dụng (CUDA, ROCm)                      | CPU thuần                               | **CPU (AVX/NEON), Apple Metal, CUDA, Vulkan** |

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
  - `_0` / `_1`: Các thuật toán làm tròn truyền thống đời đầu.
  - `_K`: **K-Quants** (lượng tử hóa phân cụm k-means, phân bổ số bit linh hoạt giữa các lớp attention và feed-forward).
  - `IQ`: **I-Matrix (Importance Matrix)**: sử dụng ma trận trọng số quan trọng dựa trên tập dữ liệu kích hoạt thực tế để nén sâu (1-3 bit) mà vẫn hạn chế tối đa suy giảm độ thông minh.
- **Quy mô kích thước (`_XS`, `_S`, `_M`, `_L`):** Extra Small, Small, Medium, Large (chỉ định các tensor quan trọng bên trong được giữ ở mức bit cao hơn).

### Bảng tổng hợp các chuẩn Quant

| Loại Quant              | Mức độ suy giảm chất lượng (Perplexity) | Dung lượng so với FP16 | Lời khuyên & Mục đích sử dụng                                                           |
|:------------------------|:----------------------------------------|:-----------------------|:----------------------------------------------------------------------------------------|
| **`Q8_0`**              | Gần như bằng 0 (~0%)                    | ~55% dung lượng gốc    | Giữ nguyên độ chính xác; dùng làm chuẩn đối sánh.                                       |
| **`Q6_K`**              | Cực kỳ thấp (<0.1%)                     | ~45% dung lượng gốc    | Tác vụ đòi hỏi độ chính xác cao (viết code, toán học).                                  |
| **`Q5_K_M`**            | Rất thấp (~0.1-0.2%)                    | ~38% dung lượng gốc    | Lựa chọn tuyệt vời khi dung lượng RAM/VRAM cho phép.                                    |
| **`Q4_K_M`**            | Thấp (~0.3-0.5%)                        | ~30% dung lượng gốc    | 🏆 **Điểm ngọt quốc dân (Sweet Spot)**: Cân bằng hoàn hảo giữa tốc độ và độ thông minh. |
| **`Q4_K_S`**            | Vừa phải                                | ~28% dung lượng gốc    | Nhẹ hơn bản M một chút, đánh đổi một phần nhỏ chất lượng.                               |
| **`IQ3_M` / `IQ3_S`**   | Trung bình                              | ~23% dung lượng gốc    | Khuyên dùng khi VRAM eo hẹp; vượt trội hơn hẳn Q3 cũ.                                   |
| **`IQ2_M` / `IQ2_XXS`** | Rõ rệt                                  | ~15-18% dung lượng gốc | Nén tối đa để cố gắng nhét mô hình 70B vào RAM 16GB-24GB.                               |

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

- **[LM Studio](https://lmstudio.ai/)** — Ứng dụng desktop hàng đầu để tìm kiếm mô hình GGUF trên Hugging Face, tải về bằng một cú click, chat và mở local OpenAI API server.
- **[Jan.ai](https://github.com/janhq/jan)** — Giải pháp thay thế ChatGPT mã nguồn mở chạy offline 100% trên máy tính với giao diện tinh tế.
- **[Ollama](https://github.com/ollama/ollama)** — Công cụ CLI và background daemon phổ biến nhất để quản lý và chạy mô hình GGUF thông qua Modelfile và REST API.
- **[Open WebUI](https://github.com/open-webui/open-webui)** — Giao diện web UI tự host giàu tính năng nhất, tích hợp sâu với Ollama, hỗ trợ RAG và quản lý nhiều người dùng.
- **[llamafile](https://github.com/Mozilla-Ocho/llamafile)** — Dự án kết hợp `llama.cpp` và Cosmopolitan Libc thành một file nhị phân duy nhất chạy được trên Windows, macOS, Linux mà không cần cài đặt bất kỳ dependency nào.
- **[Msty](https://msty.app/)** — Ứng dụng desktop bảo mật, hỗ trợ so sánh song song nhiều mô hình và quản lý thư viện prompt.
- **[Chatbox](https://chatboxai.app/)** — Client đa nền tảng kết nối trực tiếp đến các backend llama.cpp / Ollama nội bộ.
- **[Backyard AI](https://backyard.ai/)** (tên cũ Faraday) — Tối ưu hóa chuyên sâu cho các nhân vật AI hội thoại và nhập vai (roleplay) ngoại tuyến.
- **[AnythingLLM](https://github.com/Mintplex-Labs/anything-llm)** — Ứng dụng tích hợp RAG mạnh mẽ, quản lý tài liệu doanh nghiệp và chạy GGUF cục bộ.
- **[GPT4All](https://github.com/nomic-ai/gpt4all)** — Hệ sinh thái mã nguồn mở tập trung vào quyền riêng tư trên CPU và GPU phổ thông.

---

## ⚙️ Inference Engines & Serving

Các engine hiệu năng cao dùng để triển khai và phục vụ (serving) mô hình ở quy mô lớn:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — Cội nguồn của toàn bộ hệ sinh thái GGUF do Georgi Gerganov khởi xướng. Viết bằng C/C++, tăng tốc tối đa trên CUDA, Metal (Apple Silicon), ROCm, SYCL, Vulkan và OpenCL.
- **[vLLM](https://github.com/vllm-project/vllm)** — Engine serving thông lượng cao hỗ trợ nạp trực tiếp trọng số GGUF kết hợp cùng thuật toán PagedAttention.
- **[SGLang](https://github.com/sgl-project/sglang)** — Framework serving tốc độ cao cho các luồng xử lý prompt phức tạp, hỗ trợ nạp trọng số GGUF.
- **[Aphrodite Engine](https://github.com/PygmalionAI/aphrodite-engine)** — Engine inference hiệu năng cao dựa trên vLLM, chuyên dụng cho các tác vụ sáng tạo nội dung.
- **[Kobold.cpp](https://github.com/LostRuins/koboldcpp)** — Server AI all-in-one tích hợp llama.cpp cùng giao diện web phong phú của KoboldAI.
- **[LocalAI](https://github.com/mudler/LocalAI)** — REST API thay thế tương thích hoàn toàn chuẩn OpenAI để phục vụ suy luận cục bộ và phân tán.
- **[Tabby](https://github.com/TabbyML/tabby)** — Trợ lý viết code tự host, hoạt động tương tự GitHub Copilot với backend GGUF.

---

## 📱 Mobile & Edge AI

Triển khai mô hình GGUF trên điện thoại di động, máy tính bảng và chip nhúng:

- **[PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai)** — Ứng dụng mã nguồn mở tuyệt đẹp chạy trực tiếp các mô hình GGUF trên iPhone, iPad và điện thoại Android.
- **[llama.cpp on iOS](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.swiftui)** — Ví dụ mẫu SwiftUI chính thức khai thác Metal trên chip Apple A-series/M-series.
- **[MLC-LLM](https://github.com/mlc-ai/mlc-llm)** — Nền tảng biên dịch và tăng tốc đa nền tảng cho thiết bị di động và WebGPU.
- **[llama.cpp trên Raspberry Pi](https://github.com/ggml-org/llama.cpp/discussions)** — Hướng dẫn và benchmark chạy các model nhỏ 1B-3B trên Raspberry Pi 4/5.

---

## 🛠️ Công cụ Quantization & Quản lý Model

Các tiện ích chuyển đổi, phân tích metadata và nén trọng số:

- **[llama.app](https://llama.app/)** — Công cụ web tương tác trực quan để đọc metadata, cấu trúc tensor và tính toán dung lượng VRAM cần thiết.
- **[convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)** — Script chính thức chuyển đổi từ Hugging Face PyTorch/Safetensors sang định dạng GGUF.
- **[ik_llama.cpp](https://github.com/Thireus/ik_llama.cpp)** — Bộ script và hướng dẫn nâng cao về lượng tử hóa ma trận quan trọng (imatrix) do Thireus và Kalomaze phát triển.
- **[AutoGGUF](https://github.com/svjack/AutoGGUF)** — Giao diện người dùng tự động hóa toàn bộ quy trình tải, chuyển đổi và quantize model.
- **[gguf-parser](https://github.com/gpustack/gguf-parser)** — Công cụ CLI và thư viện Go/C/WASM tốc độ cao để phân tích cấu trúc file GGUF.
- **[Unsloth](https://github.com/unslothai/unsloth)** — Thư viện fine-tune siêu tốc hỗ trợ xuất thẳng model sau huấn luyện sang GGUF 4-bit / 8-bit.

---

## 📚 SDKs & Thư viện lập trình

Thư viện tích hợp GGUF vào các dự án phần mềm theo từng ngôn ngữ:

### Python
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** — Bindings Python hoàn chỉnh cho llama.cpp, tích hợp sẵn web server tương thích chuẩn OpenAI.
- **[gguf-py](https://pypi.org/project/gguf/)** — Package Python chính thức từ ggml-org để đọc và ghi cấu trúc nhị phân GGUF.

### JavaScript & TypeScript
- **[@huggingface/gguf](https://github.com/huggingface/huggingface.js/tree/main/packages/gguf)** — Đọc siêu dữ liệu và tensor header của file GGUF từ xa (remote) hoặc nội bộ mà không cần tải toàn bộ file về máy.
- **[node-llama-cpp](https://github.com/withcatai/node-llama-cpp)** — Thư viện native C++ binding mạnh mẽ cho Node.js và Bun hỗ trợ TypeScript hoàn chỉnh.
- **[Web-LLM](https://github.com/mlc-ai/web-llm)** — Chạy suy luận trực tiếp trong trình duyệt web bằng WebGPU.

### Rust
- **[candle](https://github.com/huggingface/candle)** — Framework Machine Learning tinh gọn của Hugging Face viết bằng Rust, hỗ trợ đọc và chạy mô hình GGUF.
- **[llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)** — Rust bindings an toàn cho llama.cpp.

### Go
- **[go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)** — Go bindings tương tác với llama.cpp thông qua CGo.

### .NET / C#
- **[LLamaSharp](https://github.com/SciSharp/LLamaSharp)** — C# / .NET bindings đa nền tảng để chạy llama.cpp trên Windows, Linux và macOS.

### Swift
- **[llama.cpp Swift](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.swiftui)** — Thư viện Swift package chính thức cho hệ sinh thái Apple.

---

## 🌐 Kho tải Model & Top Quantizers

Địa chỉ tìm kiếm và tải về các mô hình GGUF chuẩn chất lượng cao:

- **[Bộ lọc thư viện GGUF trên Hugging Face](https://huggingface.co/models?library=gguf)** — Danh mục chính thức với hơn 200.000+ mô hình GGUF sẵn sàng tải về.
- **[bartowski](https://huggingface.co/bartowski)** — Quantizer cộng đồng uy tín nhất hiện nay, liên tục cập nhật các bản quantize k-quant và imatrix chất lượng cao ngay khi có model mới.
- **[MaziyarPanahi](https://huggingface.co/MaziyarPanahi)** — Đóng góp số lượng lớn mô hình chuyên biệt về coding, suy luận logic và đa ngôn ngữ.
- **[mradermacher](https://huggingface.co/mradermacher)** — Vận hành hệ thống quantize tự động quy mô lớn với nhiều kiến trúc thử nghiệm độc đáo.
- **[TheBloke](https://huggingface.co/TheBloke)** — Tượng đài lịch sử tiên phong định hình làn sóng phân phối GGML/GGUF trong giai đoạn đầu của mã nguồn mở.
- **[Unsloth](https://huggingface.co/unsloth)** — Kênh phát hành chính thức các mô hình fine-tune và tối ưu động.

---

## 📖 Tài liệu chuyên sâu & Thông số kỹ thuật

Tài liệu kỹ thuật nền tảng để nghiên cứu sâu về kiến trúc GGUF:

- **[Tài liệu đặc tả kỹ thuật GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)** — Tài liệu chính thức quy định cấu trúc header, metadata key-value và căn chỉnh byte (data alignment).
- **[Tài liệu Hugging Face GGUF Hub](https://huggingface.co/docs/hub/gguf)** — Hướng dẫn chi tiết về cách Hugging Face tích hợp parser và viewer cho file GGUF.
- **[IBM: Phân biệt GGUF và GGML](https://www.ibm.com/think/topics/gguf-versus-ggml)** — Bài phân tích kiến trúc chuyên sâu so sánh sự khác biệt giữa hai thế hệ định dạng.
- **[Kalomaze: Đề xuất kỹ thuật về Importance Matrix](https://github.com/ggerganov/llama.cpp/pull/4861)** — Pull request lịch sử giải thích cơ sở toán học của kỹ thuật nén ma trận quan trọng.

---

## 🤝 Đóng góp (Contributing)

Mọi đóng góp từ cộng đồng đều được hoan nghênh! Hãy tuân theo các nguyên tắc sau:
1. Đảm bảo công cụ hoặc dự án vẫn đang được duy trì tích cực và liên quan trực tiếp đến hệ sinh thái GGUF.
2. Kiểm tra danh sách hiện có để tránh trùng lặp.
3. Gửi Pull Request với mô tả rõ ràng và sắp xếp theo thứ tự bảng chữ cái trong danh mục tương ứng.

Xem chi tiết hướng dẫn tại [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 Bản quyền (License)

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-0.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

Theo mức độ tối đa được luật pháp cho phép, các tác giả từ bỏ mọi quyền tác giả và quyền liên quan đối với công trình này.

