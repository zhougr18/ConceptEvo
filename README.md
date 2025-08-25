# ConceptEvo

ConceptEvo is a novel framework for **multimodal intent recognition**, introducing an MLLM-driven concept evolution module and a dual-path feature enhancement strategy. It establishes interpretable semantic bridges between multimodal inputs and high-level intents, yielding robust and transparent multimodal understanding.

## 1. Installation

We recommend using **Anaconda** to create the Python environment:

```
conda create --name conceptevo python=3.9
```

Install all required libraries:

```
pip install -r requirements.txt
```



## 2. Data Preparation

The datasets used in this work can be downloaded from the following links:

- [MIntRec](https://drive.google.com/file/d/16f1SOamp_hRuqRqH37eophnSQr1Dl2_w/view?usp=sharing)
- [MIntRec2.0](https://drive.google.com/drive/folders/1W-z8kMOA1TaB3pE4rk4ZpkBQLvODMQ_Y?usp=sharing)
- [MELD-DA](https://drive.google.com/file/d/1Pn-Tqok36goVdJtuxzx4fsEP0aVjeKRb/view?usp=sharing)

After downloading, place the datasets under the data/ directory:

```
data/
 ├── MIntRec/
 ├── MIntRec2.0/
 └── MELD-DA/
```



## 3. Usage

You can evaluate the performance of ConceptEvo on the datasets using the provided scripts:

```
sh examples/run_ConceptEvo.sh
```