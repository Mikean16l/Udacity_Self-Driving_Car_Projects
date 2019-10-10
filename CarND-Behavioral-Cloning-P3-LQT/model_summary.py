from keras.models import load_model
model = load_model('model.h5')
model.summary()


python drive.py model.h5
python drive.py model.h5 run1
python video.py run1 --fps 48