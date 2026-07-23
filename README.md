## TLogicNet

------------------------------
## 📌 1. Overview

A collaborative framework of logic rule-based model and neural network-based model for Temporal Knowledge Graph Completion.

### 🙏 Acknowledgments
Our codebase is partially built upon the following open-source projects. We sincerely thank the authors for their excellent work and for sharing their code:

* TLogic - Used the rule generation code for the logical rule module.
* ECEformer - Used as neural network module.

------------------------------
## ⚙️ 2. Installation

### Prerequisites

* Ubuntu 22.04 / 24.04
* Python >= 3.8
* CUDA >= 11.8 (if using GPU acceleration)
* java runtime environment >= 11

### Setup Environment
Clone this repository and install the required dependencies via requirements.txt:

```
pip install -r requirements.txt
```

Follow [our fork of ECEformer](git@github.com:LeetJoe/TKGElib.git) to install the TKGE package.

The source code of `mln/mln.jar` see [here](#todo)(coming soon).

------------------------------
## 🚀 3. Running

```
sh exp_tece.sh
```

Detailed configuration please refer to the content of `exp_tece.sh`.
