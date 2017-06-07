import os
import json
from pydoc import locate
import numpy as np

from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img, img_to_array

import logging
logging.basicConfig(level=logging.CRITICAL)

from ctk_cli import CLIArgumentParser  # noqa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(args):

    # get pretrained model
    Network = locate('keras.applications.' + args.model)
    model = Network(weights="imagenet")

    # set input shape and preprocessing function
    if args.model in ('InceptionV3', 'Xception'):

        input_shape = (299, 299)
        preprocess_input = locate('keras.applications.inception_v3.'
                                  'preprocess_input')
    else:

        input_shape = (224, 224)
        preprocess_input = imagenet_utils.preprocess_input

    # load image
    im = load_img(args.inputImageFile, target_size=input_shape)
    im = img_to_array(im)
    im = np.expand_dims(im, axis=0)

    # preprocess
    im = preprocess_input(im)

    # classify
    pred = imagenet_utils.decode_predictions(model.predict(im))[0]

    # write result
    res = []

    for (class_id, class_name, class_prob) in pred:

        print class_name + ' : ' + str(class_prob)
        res.append([class_name, float(class_prob)])

    with open(args.outputClassificationFile, 'w') as f:
        f.write(json.dumps(res, indent=1))

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
