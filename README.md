# BERT classifier for Offensive Language Detection task


### Description
----------

In this code we make use of dataset called OLIDv1.0

How to:
-------

#### First
install the requirements
```console
❱❱❱ pip install -r requirements.txt
```
Now we are good to go !!

#### Fine-tuning phase
just run the below command
```console
❱❱❱ python train.py --gpu --data_path data/ --save_path save/ --lr 5e-5 --batch_size 32 --epochs 4 --plot_path save/plot/ --bert_model bert-base-cased
```

#### Plots
![Loss](https://github.com/soroushjavdan/OffensiveBertClassifier/blob/master/save/plot.png?raw=true)
![Loss'](https://github.com/soroushjavdan/OffensiveBertClassifier/blob/master/save/plot2.png?raw=true)

