from tensorflow import keras
from skimage import io
model = keras.models.load_model('validated_model.hdf5')

test_image = io.imread("../data/processed/27-OH-CTL-data_Fib-control_CTL_uptake_24h_RhDex_Bodipy_11032020_CTL_02.tif")
test_input = test_image[8].reshape(1,512,512,1)
print(test_input.shape)
prediction = model.predict(test_input)
out = prediction.reshape(512,512)
io.imsave("predicted3.png",out)
