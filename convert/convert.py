import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras

import math

from pytorchcv.model_provider import get_model as ptcv_get_model

from keras import backend as K
import tensorflow as tf


SIZE = 224
MODEL = 'sqnxt23v5_w2'

model = ptcv_get_model('sqnxt23v5_w2', pretrained=True)
model.eval()

input_np = np.random.uniform(0, 1, (1, 3, SIZE, SIZE))
input_var = Variable(torch.FloatTensor(input_np))
output = model(input_var)

k_model = pytorch_to_keras(model, input_var, (3, SIZE, SIZE), verbose=True, name_policy='renumerate')

# Check model
pytorch_output = output.data.numpy()
keras_output = k_model.predict(input_np)
error = np.max(pytorch_output - keras_output)
print('Error: {0}'.format(error))


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


# Create, compile and train model...
frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in k_model.outputs])

tf.train.write_graph(frozen_graph, ".", "saved_model.pb", as_text=False)

# Show summary
print(k_model.summary())
print([out.op.name for out in k_model.outputs])
