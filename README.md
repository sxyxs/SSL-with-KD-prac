This is an deploying Semi-supervised Mnist classification via KD & EMA(Exponential Moving Average). 
In here, we use 100labelled data for each numbers and 20k unlabelled data to train a model. 
Also, use that 21k train data with labels to do supervised learning and compare the difference. 
The test dataset will for all the 10k Mnist dataset.

<ol> 
    <li>Train a supervied model through 21k labelled data</li> 
    <li>Train a supervied model through 1k labelled data</li> 
    <li>Use 1k model to predict pseudo label for rest of 20k data</li> 
    <li>Use 1ksup model as Teacher, use 20k pesudo label as ground truth to train a student model</li> 
    <li>Add EMA</li>
</ol>


sup 21k CNN 11epoches batchsize=30 Validation Loss: 0.036003, Accuracy: 0.990100
