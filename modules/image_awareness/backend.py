import os
import cv2
import numpy as np
from utils import ImageOpener


PATH = "modules/image_awareness/"
KERAS_MODEL = "EV-classify-1605309902.tflite"
INPUT_SIZE = 299


class ClassPredictor(ImageOpener):
    initialized = False

    def __init__(self):
        if ClassPredictor.initialized:
            return
        import tflite_runtime.interpreter as tflite
        ClassPredictor.SIZE_X = INPUT_SIZE
        ClassPredictor.SIZE_Y = INPUT_SIZE
        ClassPredictor._model = tflite.Interpreter(model_path=os.path.join(PATH, KERAS_MODEL))
        ClassPredictor._model.allocate_tensors()
        ClassPredictor.labels = [
            "Eevee",
            "Espeon",
            "Flareon",
            "Glaceon",
            "Jolteon",
            "Leafeon",
            "Sylveon",
            "Umbreon",
            "Vaporeon"
        ]
        ClassPredictor.initialized = True

    def predict(self):
        # self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB) # BRUH
        cv2.imshow('img', self._image)
        cv2.waitKey(0)

        input_ = np.asarray([self.get_img()]).astype(np.float32)
        # pd = ClassPredictor._model.predict(input_)[0]

        x = ClassPredictor._model
        x.set_tensor(0, input_)
        x.invoke()
        output_data = x.get_tensor(182)[0]
        return output_data

    def most_likely(self, pd, threshold = 0.6):
        dom = np.argmax(pd)
        if pd[dom] < threshold:
            return "I don't know"
        predicted = ClassPredictor.labels[dom]
        return predicted
        