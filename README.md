# LLL Pretrain Adapter
## Lifelong Language Learning Using Pretrained Adapters (KCC 2022)
This paper was submitted to the Korea Computer Congress (KCC), a conference hosted by the Korea Institute of Information Scientists and Engineers (KIISE).
### Author
 * [@sryndevk](https://github.com/sryndevk)

### Used data
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
Recently, various transformer-based language models have shown good performance in various NLP tasks. However, most of these models tend to learn well only for a specific task, and catastrophic forgetting occurs, which deteriorates the performance of the previous task when learning various tasks throughout life as used in the real world. To solve this problem, several transformer-based lifelong language learning techniques have been proposed. In this paper, we study lifelong language learning using adapter module based on transformer. It shows that the performance is better than the existing transformer-based lifelong language learning technique, and it shows that the performance is better than when only the adapter is used.

### Experiments
If an adapter is used for the existing Transformer-based pre-learning model, good performance can be achieved in that past knowledge is not forgotten, but it is not very efficient in terms of using previously learned knowledge for this task. Therefore, the adapter pre-learning step is added to achieve better performance than when only the adapter of the current task is trained by using the results learned from the previous adapter in the current task. We propose a technique that actively utilizes the adapter of the previous task that has been learned with past knowledge.  
Put the input EMB of this task into a predefined network with the same structure as the adapter. The target of this network performs training by designating the input of this task as the output of the first adapter that comes out when it is put into the model trained by the previous task.  
If learning proceeds in this way, the network will learn to become the first adapter, but it will not be able to follow the adapter perfectly. Therefore, this network can pre-learning in a direction that takes only the direction learned by the first adapter of the previous task.  
This means that the knowledge learned in the previous task will be able to be acquired to some extent when learning this task.  
### Model Architecture
![image](https://github.com/siryuon/LLL_Pretrain_Adapter/blob/28c0635d07bf2c72877e615e643c525db0723591/images/model.png)

### Result Table
![image](https://github.com/siryuon/LLL_Pretrain_Adapter/blob/28c0635d07bf2c72877e615e643c525db0723591/images/result.png)

### Conclusion and To Do
 * In this paper, it was shown that using the newly defined adapter pre-learning stage outperformed the existing lifelong language learning baselines. In addition, the performance of the fine-tuning method using the previously proposed adapter was exceeded, and the knowledge learned in the previous task was used to learn the current task, thereby proving the utility of knowledge accumulation.
 * Although this paper tested only one data stream, it is expected that it will show good performance for other data streams as the corresponding data stream contains natural language processing tasks for various purposes.
 * In addition, instead of using only one adapter of the previous task, it is better to study by taking all the previous adapters into consideration and to study how to better utilize the information inherent in each adapter, and to learn by bringing the learning direction of the previous task. A clear analysis of the reasons for the

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
