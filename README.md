# LLL Pretrain Adapter
## 사전학습 어댑터를 사용한 평생 언어 학습 (KCC 2022)
본 논문은 한국정보과학회(KIISE)가 주최하는 한국컴퓨터학술대회 2022(KCC 2022)에 제출되었습니다.

### 저자
 * [@lnyxzdevk](https://github.com/lnyxzdevk)

### 사용 데이터
 * [StandardStream](https://github.com/AmanDaVinci/lifelong-learning)
     * BoolQ: Question Answering
     * UDPOS: POS Tagging
     * WiC: Word-in-Context Classification
     * FewRel: Relation Extraction
     * Yelp Reviews: Sentiment Analysis

### Requirements
  * python == 3.7
  * adapter-transformers == 3.0.0
  * datasets == 2.0.0
  * huggingface-hub == 0.5.1
  * pandas == 1.3.5
  * Pillow == 9.0.1
  * pyarrow == 7.0.0
  * scikit-learn == 1.0.2
  * sacremoses == 0.0.49
  * tokenizers == 0.11.6
  * tqdm == 4.64.0
  * torch == 1.11.0

### Abstract
최근 다양한 transformer 기반 언어 모델들이 다양한 NLP 태스크에서 좋은 성능을 보여 주었다. 하지만 이런 모델들은 대부분 특정 한 태스크에 대해서만 학습이 잘 되는 경향이 있고, 실제 세계에서 사용되는 것처럼 연속적으로 다양한 태스크를 학습할 경우 이전 태스크에 대한 성능이 떨어지는 catastrophic forgetting 현상이 발생한다. 이러한 문제를 해결하기 위해 여러 transformer 기반 연속 언어 학습 기법들이 제안되었다. 본 논문에서는 transformer를 기반으로 adapter module을 활용한 연속 언어 학습을 연구한다. 기존의 transformer 기반 연속 언어 학습 기법보다 좋은 성능이 나오는 것을 보이고, adapter만 사용했을 때 보다도 좋은 성능이 나오는 것을 보인다.
### Experiments
기존 Transformer 기반 사전학습 모델에 어댑터를 사용하면 과거의 지식을 잊지 않는 다는 점에서는 좋은 성능을 낼 수 있을 지 모르지만, 이전에 학습한 지식을 이번 태스크에 활용하는 측면에서는 별다른 효율을 얻지 못한다. 따라서 이전 어댑터에서 학습한 결과를 현재 태스크에서 활용함으로써 현재 태스크의 어댑터만 학습했을 때보다 더 좋은 성능을 내기 위한 어댑터 사전학습 단계를 추가해 과거 지식으로 학습이 완료된 이전 태스크의 어댑터를 적극적으로 활용하는 기법을 제안한다.
이번 태스크의 입력 EMB를 어댑터와 동일한 구조를 가지는 사전 정의된 네트워크 〖 DN〗_T^1에 넣는다. 이 〖 DN〗_T^1의 target은 이번 태스크의 입력을 이전 태스크로 학습된 모델에 넣었을 때 나오는 첫 번째 어댑터 A_(T-1)의 출력 O_(T-1)^1로 지정하여 학습을 수행한다. 이렇게 학습이 진행된다면 〖 DN〗_T^1은 A_(T-1)가 될 수 있도록 학습이 진행되지만, 완벽하게 A_(T-1)를 따라갈 수는 없게 된다. 따라서, 〖 DN〗_T^1은A_(T-1)가 학습한 방향성만 가져가는 방향으로 사전 학습을 할 수 있게 되고, 이는 곧 이전 태스크에서 학습한 지식을 이번 태스크를 학습할 때 어느정도 얻을 수 있게 된다는 뜻이 된다.
 
### Model Architecture
![image](https://github.com/siryuon/LLL_Pretrain_Adapter/blob/28c0635d07bf2c72877e615e643c525db0723591/images/model.png)

### Result Table
![image](https://github.com/siryuon/LLL_Pretrain_Adapter/blob/28c0635d07bf2c72877e615e643c525db0723591/images/result.png)
- Finetune: 다섯가지 데이터를 순차적으로 학습
- Finetune + A: Finetune 방식에 어댑터 적용 (최종적으로는 다섯 가지 어댑터 생성)
- STL: Single Task Learning, 한 데이터만 학습
- STL + A: STL 방식에 어댑터 적용

### Conclusion and To Do
 * 본 논문을 통해 새롭게 정의한 어댑터 사전학습 단계를 활용하는 것이 기존 평생 언어 학습 baseline들의 성능을 능가하는 것을 보였다. 또한, 기존에 제안된 어댑터를 사용한 fine-tuning방식의 성능도 능가하여 이전 태스크에서 학습했던 지식을 현재 태스크를 학습할 때 활용함으로써 지식 축적의 효용성을 입증하였다.
 * 비록 본 논문에서는 한 가지 데이터 스트림에 대해서만 실험하였지만, 해당 데이터 스트림이 다양한 목적의 자연 언어 처리 태스크를 담고 있으므로 다른 데이터 스트림에 대해서도 좋은 성능을 보일 것이라고 기대된다.
 * 추가적으로, 이전 태스크의 어댑터 하나만 활용하는 것이 아닌, 이전의 모든 어댑터들을 전부 고려해 각 어댑터에 내재된 정보들을 더 잘 활용할 수 있는 방법에 대한 연구와 이전 태스크의 학습 방향성을 가져와서 학습하는 것이 더 좋은 성능을 보이는 이유에 대한 명확한 분석이 필요하다.

### References
 * Y. Huang et al. Continual learning for text classification with information disentanglement based regularization. In Proceedings of the North American Chapter of the Association for Computational Linguistics (NAACL). 2021.
 * X. Jin et al., Learn Continually, generalize rapidly: Lifelong knowledge accumulation for few-shot learning. In Proceedings of the Empirical Methods in Natural Language Processing (EMNLP). 2021.
 * F. K. Sun et al. LAMOL: Language Modeling for Lifelong Language Learning. In Proceedings of the International Conference on Learning Representations (ICLR). 2020.
 * A. Vaswani et al. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS). 2017.
 * J. Devlin et al. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the North American Chapter of the Association for Computational Linguistics (NAACL). 2019.
 * A. Radford et al. Language models are unsupervised multitask learners. Technical report. OpenAI. 2019.
 * A. Hussain et al. Towards a robust experimental framework and benchmark for lifelong language learning. In Advances in Neural Information Processing Systems (NeurIPS). 2021.
 * N. Houlsby et al. Parameter-efficient transfer learning for NLP. In Proceedings of the International Conference on Machine Learning (ICML). 2019.
 * M. Lewis et al. BART: Denoising sequence-to-sequence pretraining for natural language generation, translation, and comprehension. In Proceedings of the Association for Computational Linguistics (ACL). 2020.
