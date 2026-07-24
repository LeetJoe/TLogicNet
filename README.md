## TLogicNet

------------------------------
## 📌 1. Overview

A collaborative framework of logic rule-based model and neural network-based model for Temporal Knowledge Graph Completion.

### 🙏 Acknowledgments
Our codebase is partially built upon the following open-source projects. We sincerely thank the authors for their excellent work and for sharing their code:

* [TLogic](https://github.com/liu-yushan/TLogic) - Used the rule generation code for the logical rule module.
* [ECEformer](https://github.com/seeyourmind/TKGElib) - Used as neural network module.

------------------------------
## ⚙️ 2. Installation

### Prerequisites

* Ubuntu 22.04 / 24.04
* Python >= 3.8
* CUDA >= 11.6 (if using GPU acceleration)
* java runtime environment >= 11

### Setup Environment
1). Clone this repository and install the required dependencies via requirements.txt:

```
pip install -r requirements.txt
```

2). Follow [our fork of ECEformer](https://github.com/LeetJoe/TKGElib) to install the TKGE package.

3). Download the jar archive [Apache Commons CLI](https://commons.apache.org/cli/download_cli.cgi) (version >= 1.9.0) and put it into the `mln` directory.

4). The source code of `mln/mln.jar` see [here](#todo)(coming soon).

------------------------------
## 🚀 3. Running

```
sh exp_tece.sh
```

Detailed configuration please refer to the content of `exp_tece.sh`.
