# brain-tumor-detection-using-mri-images

 <h4>Dataset can be obtained from:</link>https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection</link></h4>
 <ul>
<li>The script <ins>main.py</ins> consists of VGG16 pretrained model which is easily accessed using keras.The script uses the VGG16 has a image feature extractor by freezing the weights i.e imagenet weights are used.The feature ouput is passed through Dense and output is obtained.</li>
<li>The  script <ins>predict.py</ins> uses a pretrained model that is saved after running  the <ins>main.py</ins> script and the result output is as shown in the below image</li>
 </ul>

     
![image](https://github.com/nishalk01/brain-tumor-detection-using-mri-images/blob/master/s.jpg)
<h1>How to run </h1>
<ul>
  <li>Make sure u have opencv-python,numpy,keras installed in your machine</li>
  <li>Keras will download the VGG16 model(589MB) ,if the model is not present in your system so make sure you have internet connectivity</li>
 <li>Change the directory path to the path where your dataset is present in the <ins>main.py</ins></li>
  <li>Finally run the command <ins>python main.py</ins></li>
 <li>After training change the image file path and run the command <ins>python predict.py</ins></li>
  </ul>
 <b>References:</b>
 <ul>
 <li><link>https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/</link></li>
 <li><link>https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/</link></li> 
  
