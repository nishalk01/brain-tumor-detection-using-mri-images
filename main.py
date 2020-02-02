from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense,Flatten
from keras.preprocessing.image import ImageDataGenerator
BS=32
path="/home/nishal/Downloads/brain_tumor_dataset/folder"
model=VGG16(include_top=False,input_shape=(224,224,3))
for layer in model.layers:
    layer.trainable=False
flat=Flatten()(model.outputs)
fin_dense=Dense(1024,activation='relu')(flat)
label=Dense(2,activation='softmax')(fin_dense)
model=Model(inputs=model.inputs,outputs=label)
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
data_augment = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")
train_generator=data_augment.flow_from_directory(path,target_size=(224,224),batch_size=BS)
model.summary()
model.fit_generator(train_generator,steps_per_epoch=253//BS,epochs=5)
model.save_weights('model.h5')
model_json = model.to_json()
with open('model.json', "w") as json_file:
    json_file.write(model_json)
json_file.close()
