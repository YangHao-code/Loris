# Harnessing the Power of Beta Scoring in Deep Active Learning for Multi-Label Text Classification

This project provides the official implementation of our paper:
**"Harnessing the Power of Beta Scoring in Deep Active Learning for Multi-Label Text Classification"**, accepted at **AAAI 2024**.

We present a multi-label text classification framework enhanced with beta-based scoring functions for uncertainty estimation and active data selection. The implementation is based on Tencent’s NeuralNLP-NeuralClassifier, with several architecture modifications and experimental enhancements.

---

## Acknowledgements & Licensing

This project builds upon the neural classification architecture provided by Tencent’s **NeuralNLP‑NeuralClassifier**, which serves as the foundation for our multi-label classification system.

Tencent’s code is open-sourced under the **MIT License**, with some components under **Apache 2.0** (see `License_for_NeuralClassifier.TXT`). We have preserved all original license headers and files in the modified source code to comply with license obligations.

**Acknowledgements:**  
We sincerely thank the Tencent NeuralNLP‑NeuralClassifier team and contributors for their high-quality open-source implementation, which enabled this work.

---

## Disclaimer

This repository contains research-oriented, experimental code provided “as is,” with no warranties, express or implied. It supports the findings presented in our AAAI 2024 paper but may not fully replicate all results due to dependency versions, data configurations, and environment variability.

By using this code, you agree to the following:

- The code is provided “as is,” without warranties of any kind, express or implied.
- It is intended to support understanding of the research method and is suitable for academic and exploratory use.
- The implementation is undergoing ongoing refactoring and optimization.

---

## Technical Overview

We use the core classifier modules (e.g., encoder layers, training pipeline) from Tencent’s NeuralClassifier as a base, and introduce:

- **Beta scoring functions** for uncertainty quantification in active learning.
- **Deep active learning loops** tailored for multi-label classification.
- **Enhanced metrics**, logging, and analysis tools for multi-label evaluation.
- **Custom loss extensions** and sampling strategies for selection.

All adapted code retains appropriate licensing notices.

---

## License Summary

- **This project:** MIT License (see `LICENSE`).
- **Included components:** Derived from Tencent’s NeuralNLP‑NeuralClassifier (MIT License), with dependencies under Apache 2.0 (see `License_for_NeuralClassifier.TXT`).

By using or modifying this repository, you agree to comply with all relevant license terms.

---

## Citation

If you use this code, please cite our paper:

@inproceedings{tan2024harnessing,
  title={Harnessing the power of beta scoring in deep active learning for multi-label text classification},
  author={Tan, Wei and Nguyen, Ngoc Dang and Du, Lan and Buntine, Wray},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={14},
  pages={15240--15248},
  year={2024}
}


Paper: [Harnessing the Power of Beta Scoring in Deep Active Learning for Multi-Label Text Classification](https://ojs.aaai.org/index.php/AAAI/article/view/29447)  

You are also encouraged to acknowledge the technical base provided by Tencent’s [NeuralNLP‑NeuralClassifier](https://github.com/Tencent/NeuralNLP-NeuralClassifier) in your work.


## Configuration How to Run

1. Clone the Repository
git clone https://github.com/davidtw999/BESRA.git
cd BESRA

2. Setup Python Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Prepare Dataset
Place your dataset under the dataset/ directory.

3. Run Active Learning
Please configure the query and parameters before you run the experiment.
python train_al.py conf/train_al.json
