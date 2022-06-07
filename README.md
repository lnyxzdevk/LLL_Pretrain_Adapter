# LLL Pretrain Adapter
## Lifelong Language Learning Using Pretrained Adapters (KCC 2022)
This paper was submitted to the Korea Computer Congress (KCC), a conference hosted by the Korea Institute of Information Scientists and Engineers (KIISE).
### Author
 * Cho Moon Gi [@siryuon](https://github.com/siryuon)

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

### Model Architecture
![image](https://github.com/siryuon/LLL_Pretrain_Adapter/blob/28c0635d07bf2c72877e615e643c525db0723591/images/model.png)

### Result Table
![image](https://github.com/siryuon/LLL_Pretrain_Adapter/blob/28c0635d07bf2c72877e615e643c525db0723591/images/result.png)

### Conclusion and To Do
 * In this paper, it was shown that using the newly defined adapter pre-learning stage outperformed the existing lifelong language learning baselines. In addition, the performance of the fine-tuning method using the previously proposed adapter was exceeded, and the knowledge learned in the previous task was used to learn the current task, thereby proving the utility of knowledge accumulation.
 * Although this paper tested only one data stream, it is expected that it will show good performance for other data streams as the corresponding data stream contains natural language processing tasks for various purposes.
 * In addition, instead of using only one adapter of the previous task, it is better to study by taking all the previous adapters into consideration and to study how to better utilize the information inherent in each adapter, and to learn by bringing the learning direction of the previous task. A clear analysis of the reasons for the
