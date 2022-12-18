# Inference

Inference is performed with `train.py`. You have to change the following:

+ Name of the model: number is the name of the model. The best model is by default.
    + `number = "third_wobranches_like29_reduced"`
+ Output shape:
    + `input_shape_prediction = (768, 1280, 3)`
+ Image folder: the folder where images are found. it has to be a folder inside a folder
    + `experiment.hdr_converter.predict("./images", (input_shape_prediction[0], input_shape_prediction[1]), experiment.tflite_name_fusion, crop=False, video=False)`
    + You can select to crop the image or perform inference on a video. In that case you do not need to put videos inside a folder inside your prediction folder.
    + INference in desktop is performed by tflite interpreter which is horribly slow on that kind of CPUs but is the only why to infer with quantization.
+ Set `experiment.train_or_resume(train=True, resume=False)` to train, and `experiment.train_or_resume(train=True, resume=False)` to resume. BEware that training over a model will overwrite it.

## Change the model

You can change the model by putting the name of any of the folders at the folder ![results](H:\Projects\AIDI\GN1_AI_Driven_Game_Experience\MQITM\results\models\tf). 

NOTE: Some models require to change some parts of the network if you want to really retrain them or change output size. For example, attention models require to swap each of the attention modules. You only need to change the name of the module you want to use to `channel_attention_block`. The ![guide](![results](H:\Projects\AIDI\GN1_AI_Driven_Game_Experience\MQITM\results\test_data\guide.txt) can help you know which model is which.


