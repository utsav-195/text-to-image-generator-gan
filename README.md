# Text to Image Synthesizer using Generative Adversarial Networks
The purpose of this project is to train a generative adversarial network (GAN) to generate images from textual description of the image.
For this particular project, I have used flower images from the [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

## Data:
The data of the image description was obtained from [here](https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view).
The image caption data link was obtained from the following [github](https://github.com/zsdonghao/text-to-image).

The flowers dataset has 102 categories of flower images. Each category has 40-258 images.
The total number of flower images-description pairs used in this project is 8100.

The image descriptions were processed into character level embeddings using the 300 D GloVe embeddings. You can obtain them from [here](https://nlp.stanford.edu/projects/glove/).
I used the embeddings created on Wikipedia 2015 and Gigaword-5 data.

The images of the flowers are as below:
![Flowers](screenshots/Capture2.PNG?raw=true)

The captions are as below:
![Captions](screenshots/Capture1.PNG?raw=true)

## Architecture:
This is an experimental implementation of synthesizing images. The images are synthesized using the GAN-CLS Algorithm from the paper [Generative Adversarial Text-to-Image Synthesis](http://arxiv.org/abs/1605.05396). However, we have not used Skip-Thoughts vectors, instead, we tried the implementation using the GloVe embeddings.

![Model architecture](screenshots/Capture4.jpg)

Image Source : [Generative Adversarial Text-to-Image Synthesis](http://arxiv.org/abs/1605.05396) Paper.

Note: We have tried to implement the GAN architecture as best as we could. It is not guaranteed to be the exact same as the one in the paper.

## Implementation:
The iPython notebook has comments indicating what each cell does.
You can then execute the cells in `text_to_image_synthesizer_glove.ipynb` to see how everything works.

The notebook has 5 parts:
- Data Pre-processing:
    - Converting the images into numpy arrays based on the pre-set pixel size and storing them.
    - Converting the image description into embeddings and storing them.
    - Sotring the captions in a csv file.
- Loading and Combining:
    - Loading all the images' numpy arrays and appending it to form the image data.
    - Loading the image description embedding numpy array.
- Data modeling:
    - Creating the discrimintor and generator networks.
    - Functions for calculating the discriminator and generator loss.
    - Setting the optimizer and learning rate.
- Training:
    - Function for the train step, that creates the images, calculates the loss and adjusts the gradients.
    - Function for the train, it fetches the batch data and passes it to train_step and collects all the metrics.
- Results:
    - Function for testing the output from the generator given an input noise and a caption

## Result:
![Result1](screenshots/Capture3.PNG?raw=true)

Note: Addtional references in the iPython notebook.
