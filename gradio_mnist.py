import os
import gradio as gr
import tensorflow as tf
import numpy as np

# model_filename = "basic_fully_connected_mnist"
model_filename = "mnist_cnn"

def load_model(filename, filepath):
    model_fname = filename
    my_wd = filepath

    return(tf.keras.models.load_model(os.path.join(my_wd, model_fname)))

mnist_cnn = load_model(model_filename, os.getcwd())

def reshape_for_cnn(image):
    image = image/255.0
    image = image.reshape(28, 28, 1)
    image = (np.expand_dims(image,0)) # add a batch dim
    print(image.shape)
    return(image)

# now get the model to recognise new input images
def recognise_image(image):
    image = reshape_for_cnn(image)
    prediction = mnist_cnn.predict(image)
    return {str(i): float(prediction[0][i]) for i in range(len(prediction[0]))}

output_component = gr.outputs.Label(num_top_classes=3) # need to adjust this...
gr.Interface(fn=recognise_image, 
             inputs="sketchpad", 
             outputs=output_component,
             title="MNIST CNN Sketchpad",
             description="Draw a number 0 through 9 on the sketchpad, and click submit to see the model's predictions. Model trained on the MNIST dataset.",
             thumbnail="https://raw.githubusercontent.com/gradio-app/real-time-mnist/master/thumbnail2.png").launch();

