# Neural Network Charity Analysis

## Project Overview

The purpose of this project is to assess the performance of a deep learning model in determining good charities for a non-profit foundation to support. The model uses the TensorFlow to evaluate the charity data using neural network deep learning. 

## Results

### Data Processing

- The target for the model is whether a charity was successful in achieving their goal. 

- The features for the model are encoded versions of all columns below:

  - ```python
    APPLICATION_TYPE	AFFILIATION	CLASSIFICATION	USE_CASE	ORGANIZATION	STATUS	INCOME_AMT	SPECIAL_CONSIDERATIONS	ASK_AMT
    ```

- The 'EIN' and 'Status' columns were removed from the dataset because they have no bearing on the model. Additionally, the 'Application Type' and 'Classification' columns were binned to focus on unique data points for the model. The elbow curves for each data column are shown below:

  - ![](https://raw.githubusercontent.com/CarlS2rt/neural_network_analysis/main/images/application_elbow.png)
  - ![](https://raw.githubusercontent.com/CarlS2rt/neural_network_analysis/main/images/classification_elbow.png)

### Compiling, Training, and Evaluating the Model

- The original model was designed as follows:

  

  - ```python
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 8)                 352       
                                                                     
     dense_1 (Dense)             (None, 5)                 45        
                                                                     
     dense_2 (Dense)             (None, 1)                 6         
                                                                     
    =================================================================
    Total params: 403
    Trainable params: 403
    Non-trainable params: 0
    ```

  - These values were chosen as a good starting point based on performance of other similar models. 

- The model's performance was, however, less than optimal. The results are shown below:

  

  - ```python
    268/268 - 0s - loss: 0.8482 - accuracy: 0.5444 - 209ms/epoch - 781us/step
    Loss: 0.8481515645980835, Accuracy: 0.5443731546401978
    ```

  - With high loss and low accuracy in the original model, three additional attempts were made to improve the accuracy of the model.

#### Optimization 1:

The first optimization attempt differed from the original model only in that Keras Tuner was utilized to determine the most optimal hyperparameters for the model. 

The top three hyperparameter values were as follows:

```python
{'activation': 'tanh', 'first_units': 21, 'num_layers': 4, 'units_0': 11, 'units_1': 11, 'units_2': 11, 'units_3': 11, 'tuner/epochs': 7, 'tuner/initial_epoch': 0, 'tuner/bracket': 1, 'tuner/round': 0}
{'activation': 'tanh', 'first_units': 26, 'num_layers': 5, 'units_0': 11, 'units_1': 26, 'units_2': 26, 'units_3': 26, 'units_4': 26, 'tuner/epochs': 20, 'tuner/initial_epoch': 7, 'tuner/bracket': 1, 'tuner/round': 1, 'tuner/trial_id': '0048'}
{'activation': 'relu', 'first_units': 21, 'num_layers': 2, 'units_0': 1, 'units_1': 6, 'units_2': 16, 'units_3': 11, 'tuner/epochs': 7, 'tuner/initial_epoch': 0, 'tuner/bracket': 1, 'tuner/round': 0}
```

The results of each model using the updated hyperparameter values were much improved over the original model but still not quite to the 75% target. 

```python
268/268 - 0s - loss: 0.5511 - accuracy: 0.7364 - 334ms/epoch - 1ms/step
Loss: 0.5510514974594116, Accuracy: 0.7364431619644165
268/268 - 0s - loss: 0.5473 - accuracy: 0.7364 - 336ms/epoch - 1ms/step
Loss: 0.5473225116729736, Accuracy: 0.7364431619644165
268/268 - 0s - loss: 0.5539 - accuracy: 0.7361 - 301ms/epoch - 1ms/step
Loss: 0.553896427154541, Accuracy: 0.736093282699585
```

#### Optimization 2:

The second optimization model removed additional values that were neither targets nor features: 'Special Considerations' and 'Status'. Additionally, the binning values for 'Applications Type' and 'Classification' were increased, and 'Income Amount' was also binned to focus in on unique values in that dataset. 

The Keras Tuner was again applied to the model to find the optimal hyperparameters based on the new data preprocessing. The top three model hyperparameters are as follows:

```python
{'activation': 'relu', 'first_units': 26, 'num_layers': 2, 'units_0': 26, 'units_1': 6, 'units_2': 16, 'units_3': 6, 'units_4': 16, 'tuner/epochs': 20, 'tuner/initial_epoch': 7, 'tuner/bracket': 1, 'tuner/round': 1, 'tuner/trial_id': '0022'}
{'activation': 'relu', 'first_units': 26, 'num_layers': 3, 'units_0': 16, 'units_1': 26, 'units_2': 1, 'units_3': 6, 'units_4': 6, 'tuner/epochs': 20, 'tuner/initial_epoch': 0, 'tuner/bracket': 0, 'tuner/round': 0}
{'activation': 'relu', 'first_units': 26, 'num_layers': 2, 'units_0': 21, 'units_1': 21, 'units_2': 21, 'units_3': 16, 'units_4': 21, 'tuner/epochs': 20, 'tuner/initial_epoch': 0, 'tuner/bracket': 0, 'tuner/round': 0}
```

The results of the second optimization were slightly lower than the first optimization but still much higher than the original model. 

```python
268/268 - 0s - loss: 0.5821 - accuracy: 0.7151 - 300ms/epoch - 1ms/step
Loss: 0.5820741653442383, Accuracy: 0.7151020169258118
268/268 - 0s - loss: 0.5820 - accuracy: 0.7150 - 352ms/epoch - 1ms/step
Loss: 0.5819693207740784, Accuracy: 0.7149854302406311
268/268 - 0s - loss: 0.5797 - accuracy: 0.7139 - 333ms/epoch - 1ms/step
Loss: 0.5797211527824402, Accuracy: 0.7139358520507812
```

#### Optimization 3:

The final optimization attempt utilized slight variations on previous optimization attempts. The 'Status' values were dropped from the model, and the bin sizes were decreased for both 'Application Type' and 'Classification'. No additional values were binned or removed from the model. 

In addition to the above preprocessing of the data, the Keras Tuner was adjusted slightly to attempt to reach or exceed the 75% accuracy target. To achieve this, the minimum number of neurons was increased to 6, and the maximum epochs were lowered to 20. The results of the final attempt are below:

```python
{'activation': 'tanh', 'first_units': 16, 'num_layers': 4, 'units_0': 26, 'units_1': 6, 'units_2': 26, 'units_3': 11, 'units_4': 11, 'tuner/epochs': 7, 'tuner/initial_epoch': 0, 'tuner/bracket': 1, 'tuner/round': 0}
{'activation': 'relu', 'first_units': 11, 'num_layers': 3, 'units_0': 6, 'units_1': 21, 'units_2': 26, 'units_3': 11, 'units_4': 11, 'tuner/epochs': 20, 'tuner/initial_epoch': 0, 'tuner/bracket': 0, 'tuner/round': 0}
{'activation': 'tanh', 'first_units': 26, 'num_layers': 3, 'units_0': 21, 'units_1': 6, 'units_2': 6, 'units_3': 26, 'units_4': 6, 'tuner/epochs': 20, 'tuner/initial_epoch': 0, 'tuner/bracket': 0, 'tuner/round': 0}
```

The results of the final optimization yielded the best results overall for the model, but the target accuracy of 75% was still not achieved. 

```python
268/268 - 1s - loss: 0.5478 - accuracy: 0.7388 - 691ms/epoch - 3ms/step
Loss: 0.5478343367576599, Accuracy: 0.7387754917144775
268/268 - 1s - loss: 0.5474 - accuracy: 0.7388 - 737ms/epoch - 3ms/step
Loss: 0.5473676323890686, Accuracy: 0.7387754917144775
268/268 - 1s - loss: 0.5482 - accuracy: 0.7384 - 820ms/epoch - 3ms/step
Loss: 0.548239529132843, Accuracy: 0.7384256720542908
```



## Summary

Overall, the model's best iteration achieved 73.88% accuracy. Through several attempts with the Keras Tuner, the target threshold could not be achieved. Recommendations to get the model above the target accuracy score include additional preprocessing to remove outliers and non-relevant data from the model. Additionally, using a model other than sequential or leaving the Keras Tuner completely open to choose the model and hyperparameters could help. Additionally, it is also a possibility that more data points are needed for the model or there are not reliably strong indicators of a charity's success. 