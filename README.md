# Mertcan Gelbal

**AI/ML Engineer | Generative AI, Computer Vision & Edge AI**

Computer Engineering graduate (Marmara University) with hands-on experience in machine learning, generative AI, computer vision, and edge inference. I build reproducible AI systems end to end — from dataset preparation and model evaluation to mobile, API, and NVIDIA Jetson deployment — and document what was actually measured, including limitations.

[LinkedIn](https://www.linkedin.com/in/mertcangelbal) · [Email](mailto:gelbalmertcan@gmail.com) · [Portfolio](https://mertcan-gelbal.github.io/mertcan-gelbal-portfolio/)

---

## Featured Projects

### [Botanix — On-Device Plant Disease Detection (Mobile)](https://github.com/Mertcan-Gelbal/botanix-mobile-ai)
Farmers need field diagnosis without connectivity; this thesis project ships a MobileNetV2 classifier fully on-device in a React Native app.
**Contribution:** end-to-end mobile inference pipeline (image capture → preprocessing → TFLite inference → Turkish-localized results) with offline operation.
**Tech:** React Native 0.79 / Expo 53, TensorFlow Lite (`react-native-fast-tflite`), ONNX Runtime, TypeScript.
**Evidence-backed capability:** bundles a 38-class PlantVillage TFLite model (upstream eval accuracy 78.6%) and runs it offline on device.

### [Botanix Model Benchmark (Notebooks)](https://github.com/Mertcan-Gelbal/botanix-model-benchmark)
Which architecture actually wins on large-scale leaf-disease classification?
**Contribution:** a 7-architecture comparison (CNN baseline, ViT-B/16, Swin-B, self-supervised segmentation-head variants, from-scratch CNNs with a PlantCLEF pre-training plan) on a 105-class dataset I published on Kaggle.
**Tech:** PyTorch, timm, scikit-learn, Jupyter.
**Evidence-backed capability:** fully specified training/eval protocol (augmentation, weighted CE, cosine LR) with a documented ~374K-image dataset split.

### [Agricultural RAG & BERT Classification](https://github.com/Mertcan-Gelbal/agricultural-rag-edge-ai)
Domain question-answering for agriculture, combining text classification with retrieval-augmented generation.
**Contribution:** trained and compared BERT-family classifiers on a 6-category agricultural dataset; built a sentence-transformers RAG pipeline and Streamlit/API front ends, with Jetson deployment scripts.
**Tech:** PyTorch, Hugging Face Transformers, sentence-transformers, Streamlit, NVIDIA Jetson (JetPack 6.2).
**Evidence-backed result:** DistilBERT fine-tune reached 85.9% accuracy / 0.861 F1 on the held-out validation split (metrics file in repo).

### [Jetson Orin Setup Toolkit](https://github.com/Mertcan-Gelbal/jetson-orin-setup-toolkit)
A freshly flashed Jetson takes hours of manual setup; this toolkit does it in one command.
**Contribution:** modular, idempotent Bash toolkit (config-driven, logged, resumable) covering system update, GStreamer/OpenCV, Jetson PyTorch wheel, Docker, and a verification suite.
**Tech:** Bash, JetPack 5.x/6.x, Docker, CUDA/cuDNN tooling.
**Evidence-backed capability:** verify-only mode audits CUDA, PyTorch, OpenCV build flags, and camera pipelines on the device.

### [Health Assistant — LLM Evaluation Prototype](https://github.com/Mertcan-Gelbal/health-assistant-llm-evaluation)
How do Gemini and GPT compare on Turkish health-assistant intents?
**Contribution:** evaluation-oriented Streamlit app comparing two LLM APIs on an 8-intent, 1,250-sample synthetic dataset, with a keyless demo mode and explicit no-diagnosis policy.
**Tech:** Python, Streamlit, Google Gemini API, OpenAI API, scikit-learn.
**Evidence-backed capability:** side-by-side model comparison with intent classification; benchmark results intentionally reported only when measured.

### [HTTP Server From Scratch](https://github.com/Mertcan-Gelbal/http-server-from-scratch)
HTTP implemented directly on TCP sockets — no frameworks.
**Contribution:** threaded request handling, static file serving with MIME detection, JSON API routes, directory-traversal protection, Docker packaging with healthcheck.
**Tech:** Python (stdlib only), Docker, docker-compose.
**Evidence-backed capability:** runs as a container with a working health endpoint; protocol behavior documented with example request/response pairs.

---

## Core Engineering Areas

- **Machine Learning** — PyTorch, TensorFlow/Keras, scikit-learn, transfer learning, evaluation protocols
- **Generative AI** — Hugging Face Transformers, RAG (sentence-transformers), LLM API integration & evaluation
- **Computer Vision** — CNNs, ViT/Swin, image classification pipelines, OpenCV
- **Edge AI** — TensorFlow Lite, ONNX Runtime, NVIDIA Jetson (JetPack, CUDA, TensorRT-oriented workflows)
- **Backend & Deployment** — Python, FastAPI/Streamlit, Docker, Go (gRPC), REST/gRPC fundamentals
- **Engineering Tools** — Git/Git LFS, Linux, Bash, CI (GitHub Actions), Kaggle datasets

## Current Focus

- Completing the Botanix benchmark runs (ViT/Swin vs from-scratch CNN with segmentation heads) and publishing the measured results
- PlantCLEF-style pre-training for plant domain models
- Making Jetson edge deployment reproducible and verifiable end to end

## Contact

- Email: gelbalmertcan@gmail.com
- LinkedIn: https://www.linkedin.com/in/mertcangelbal
- Website: https://mertcan-gelbal.github.io/mertcan-gelbal-portfolio/
