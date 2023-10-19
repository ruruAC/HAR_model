# HAR_model
![image](https://github.com/ruruAC/HAR_model/assets/81458165/5b0ce27d-e5d2-4e6f-be0b-273720e5ebd8)

During system runtime, the terminal collects Human Activity Recognition (HAR) data and uploads it to the cloud. The cloud processes the received data, trains the model, and deploys the pretrained model to edge nodes. Edge nodes load the pretrained model and, when computationally feasible, receive data from the terminal for inference, returning the results to the terminal. Additionally, considering user heterogeneity, a portion of specific user data is used to retrain the model. In cases where edge nodes cannot handle the computations, the cloud platform receives user data and sends back inference results. Finally, the terminal receives the inference results and presents them visually. The overall framework is as follows:
![image  height="50"](https://github.com/ruruAC/HAR_model/assets/81458165/62ad6a31-e53b-421f-b4be-206c9f14758e)

The specific steps are as follows:

Step 1  Run HAR_model.ipynb.
Step 2  Run light_HAR.ipynb.
Step 3  Run HAR_train+.ipynb.
Step 4  Run raspberry_load_predict.py.

The demonstration capability on the ported device is [here]([https://github.com/ruruAC/HAR_model/blob/master/HAR.mp4)).
