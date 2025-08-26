#  MLLM-Driven Concept Evolution and Multimodal Feature Enhancement for Intent Recognition

This project leverages multimodal large language models (MLLMs) to drive concept evolution. It dynamically generates and refines fine-grained semantic concepts, enhancing multimodal feature representations through a dual-path mechanism: one path amplifies concept-relevant cues via semantic gating, while the other introduces informative negatives for contrastive learning. This synergistic design improves both accuracy and interpretability, achieving state-of-the-art performance on multiple intent recognition benchmarks.

All required configuration files are included, and the entire training and evaluation pipeline can be launched with a single command.

## Project Structure
ConceptEvo/
├── backbones/                          
│   └── FusionNets/
│       ├── ConceptEvo.py      
│       └── ConceptEvo_GPT.py            

├── configs/                             
│   ├── ConceptEvo_mintrec.py/                      
│   ├── ConceptEvo_mintrec2.py/                     
│   ├── ConceptEvo_meld_da.py/                 
│   ├── ConceptEvo_GPT_mintrec.py/                     
│   ├── ConceptEvo_GPT_mintrec2.py/      
│   └── ConceptEvo_GPT_meld_da.py  

├── datasets/                             
│   ├── MIntRec/                      
│   ├── MIntRec2.0/                         
│   └── MELD-DA  

├── examples/                             
│   ├── run_ConceptEvo_mintrec.sh/                      
│   ├── run_ConceptEvo_mintrec2.sh/                     
│   ├── run_ConceptEvo_meld_da.sh/                 
│   ├── run_ConceptEvo_GPT_mintrec.sh/                     
│   ├── run_ConceptEvo_GPT_mintrec2.sh/      
│   └── run_ConceptEvo_GPT_meld_da.sh          


├── methods/                          
│   ├── ConceptEvo/                                  
│   │     ├── evolution_ConceptEvo.py              
│   │     ├── losses.py
│   │     └── manager.py      
│   └── ConceptEvo_GPT/                                  
│         ├── evolution_ConceptEvo_GPT.py              
│         ├── losses.py
│         └── manager.py           
└── README.md                        


**Note:** `ConceptEvo_GPT` is a variant of `ConceptEvo`, where GPT-4 is used as the concept-driving LLM instead of Gemini-2.0. All other components remain identical.

# Model Introduction

* **`backbones/FusionNets/ConceptEvo.py`**: This file contains the implementation for the model of the ConceptEvo method which is a dual-path concept-aware feature enhancement module. It represents the core contribution and primary focus of our model designing.

*Semantic-Enhancing Path*: In this pathway, the model first obtains token-level multimodal representations through a cross-modal fusion network and computes attention distribution based on similarity with concept vectors. Subsequently, concept features are aggregated using attention weights to obtain a semantically enhanced vector for each token. To achieve an adaptive balance between original and concept-enhanced features, the model fuses them through a gating module, selectively emphasizing key concept-related information. This allows the model to more effectively capture semantically important cues in complex interactions, improving the accuracy and interpretability of intent recognition.

*Contrast-Inducing Path*: In this path, the model uses concept attention distribution to identify tokens with high semantic relevance and then masks or perturbs these high-attention regions according to a set ratio, thereby constructing comparison samples that are "missing key information." This operation forces the model to distinguish between original samples and masked samples during learning, forming a more discriminative representation space. By introducing this contrast signal during training, the model not only enhances its robustness to concept semantics but also learns clearer discriminative boundaries in cross-modal features.


* **`methods/ConceptEvo/manager.py`**: This manager file implements the ConceptEvo training and evaluation process, integrating model optimization, early stopping, and comparative learning objectives. It also combines concept initialization and iterative updates driven by Gemini-2.0 to support the implementation of concept evolution in the actual training process.
*MLLM-driven Concept Evolution*:
In our implementation, the MLLM-driven Concept Evolution module is driven by Gemini-2.0, which dynamically updates the concept set by combining discriminability and diversity feedback. First, during the initialization phase, we call Gemini-2.0 via concepts_init . Using the task context (intent categories and their representative examples) and a carefully designed prompt, we generate a set of highly task-relevant and discriminative semantic concepts. Subsequently, during the iterative update phase, after each training epoch, the model computes the discriminability score D(c) (reflecting the ability of a concept to distinguish between different intent categories) and the diversity score S(c) (reflecting the semantic redundancy between concepts) based on the prediction results. These metrics, along with the training state (loss, validation score) and historical context, are fed into concepts_evolution , where Gemini-2.0 generates an optimized concept set. This iterative process allows the model to continuously retain discriminative and diverse concepts and replace redundant or invalid ones, ensuring that the concept set remains discriminative, representative, and interpretable, providing a reliable semantic anchor for cross-modal feature enhancement.


---

# Datasets Usage
The dataset used in this project can be downloaded from the following link:  
MIntRec[Download Here](https://drive.google.com/file/d/16f1SOamp_hRuqRqH37eophnSQr1Dl2_w/view?usp=sharing)
MIntRec2.0[Download Here](https://drive.google.com/file/d/18uSswLXKUz43QAgMiCX_T5gV8EkDIy89/view?usp=drive_link)
MELD-DA[Download Here](https://drive.google.com/file/d/1Pn-Tqok36goVdJtuxzx4fsEP0aVjeKRb/view?usp=sharing)
# Training or Testing commands

You can directly start training or evaluation using the following command:
```bash
proxychains4 sh examples/run_ConceptEvo_mintrec.sh
```
You just need to change the .sh file to run various tasks on diverse datasets.


