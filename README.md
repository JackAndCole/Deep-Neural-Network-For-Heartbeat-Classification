# Code for: A hybrid method for heartbeat classification via convolutional neural networks, multilayer perceptrons and focal loss
## Dataset

[MIT-BIH Arrhythmia database](https://www.physionet.org/content/mitdb/1.0.0/)

## Usage

- Reproduce the results

> A pre-trained model is provided in the model directory. To reproduce our results, you should, first, download the MIT-BIH arrhythmia database from the above link and save it in the dataset directory. Then, execute  `preprocessing.py` to obtain the training and test dataset. After that, just run `pre-training.py` and you will get the results of our paper.

- Re-train the model

> Just replace the last step of run `pre-training.py` with `main.py`.
>
> Noted that the optimization function of Keras and TensorFlow is slightly different in different versions. Therefore, to reproduce our results, it suggests that using the same version of Keras and TensorFlow as us, in our work the version of Keras is  2.3.1 and TensorFlow is 1.15.0. In addition, Keras and TensorFlow have a certain randomness, the actual results may be somewhat floating.  

## Cite

The paper is submitted to the journal for review. 

## Email:

If you have any questions, please email to: [wtustc@mail.ustc.edu.cn](mailto:wtustc@mail.ustc.edu.cn)
