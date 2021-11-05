# Counterfactual Data Augmentation for Aspect Sentiment Classication

In this paper, we propose a counterfactual data augmentation approach for aspect sentiment classication (ASC), which is shown as below:

<p>
<img src="https://github.com/anonymity014/DASFAA2022/blob/main/model.png" width="800">
</p>

Our augmentation method consists of three steps:

- Estimation

- Selection

- Discrimination

In this repo, we'll introduce the running of two modules: **discriminator** and **classifier**.The discriminator aims to train an aspect extraction model and the sentiment classification model is to test whether the generated sentences are reasonable, while the classifier retrains the generated sentences mixed with the source sentences.
##Discriminator
### requirement 
- Python 3.6
- [Pytorch 1.1](https://pytorch.org/)
- [Allennlp](https://allennlp.org/)

Download the uncased [BERT-Base](https://drive.google.com/file/d/13I0Gj7v8lYhW5Hwmp5kxm3CTlzWZuok2/view?usp=sharing) model and unzip it in the current directory. 
### command
For each dataset, we have different parameters and need to replace different codes:

```
python absa.run_joint_span 
```

    # if dataset is Res15
    ```
    parser.add_argument("--train_file", default='Train-Test-data/rest_total_train.txt', type=str)
    parser.add_argument("--predict_file", default='Train-Test-data/rest_total_test.txt',type=str)
    parser.add_argument("--discriminator_file", default='Rest15/exchange.seg', type=str)
    parser.add_argument("--output_dir", default="out/Aug_Rest15" ,type=str)
    parser.add_argument("--selection_path", default='Rest15/Augmentation.seg',type=str)
    parser.add_argument("--source_path", default='Rest15/restaurant_train.raw',type=str)
     ```
    if dataset is Res16
    ```
    # parser.add_argument("--train_file", default='Train-Test-data/rest_total_train.txt', type=str)
    # parser.add_argument("--predict_file", default='Train-Test-data/rest_total_test.txt',type=str)
    # parser.add_argument("--discriminator_file", default='Res16/exchange.seg', type=str)
    # parser.add_argument("--output_dir", default="out/Aug_Res16" ,type=str)
    # parser.add_argument("--selection_path", default='Res16/Augmentation.seg',type=str)
    # parser.add_argument("--source_path", default='Res16/restaurant_train.raw',type=str)
    ```
    if dataset is Res14
    ```
    parser.add_argument("--train_file", default='Train-Test-data/rest_total_train.txt', type=str)
    parser.add_argument("--predict_file", default='Train-Test-data/rest_total_test.txt',type=str)
    parser.add_argument("--discriminator_file", default='Rest14/exchange.seg', type=str)
    parser.add_argument("--output_dir", default="out/Aug_Rest14" ,type=str)
    parser.add_argument("--selection_path", default='Rest14/Augmentation.seg',type=str)
    parser.add_argument("--source_path", default='Rest14/Restaurants_Train.xml.seg',type=str)
    ```
    if dataset is Lap14
    ```
    # parser.add_argument("--train_file", default='Train-Test-data/laptop14_train.txt', type=str)
    # parser.add_argument("--predict_file", default='Train-Test-data/laptop14_test.txt',type=str)
    # parser.add_argument("--discriminator_file", default='Lap14/exchange.seg', type=str)
    # parser.add_argument("--output_dir", default="out/Aug_Lap14" ,type=str)
    # parser.add_argument("--selection_path", default='Lap14/Augmentation.seg',type=str)
    # parser.add_argument("--source_path", default='Lap14/Laptops_Train.xml.seg',type=str)
    ```
The augmentation sentence will appear in out/.

## Classifier
### requirement 
- pytorch 1.1
- numpy >= 1.13.3
- sklearn
- python >= 3.6
- transformers

For non-BERT-based models, [GloVe pre-trained](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors) word vectors are required, please refer to data_utils.py for more detail.

For BERT-based models, Download the uncased [BERT-Base](https://drive.google.com/file/d/13I0Gj7v8lYhW5Hwmp5kxm3CTlzWZuok2/view?usp=sharing) model and unzip it in the current directory. 
### Command

            python train_k_fold.py
                  --dataset xx\
                  --ratio xx\ 
                  --num_epoch xx\
                  --batch_size xx\
                  --learning_rate xx\
                  --model_name xx\
                  --train_file xx\
                  --aug_file xx\
                  --test_file xx\
                  --result_file xx\
The range of parameters in this paper is as follows:

	```
	the dataset:['Lap14','Res14','Res15','Res16']
	the batch size:[16,32,64,128]
	the ratio:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.9,1.0]
	the learning_rate:[1e-5,3e-5,5e-5,1e-3,3e-3,5e-3]
	the model_name:['aen_bert','bert-spc','aen','tnet_lf'']
	the num_epoch :[10,20,30,40,50]





