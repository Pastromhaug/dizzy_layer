import tensorflow as tf
import numpy as np
from layers.dizzyRNNCellOpt import DizzyRNNCellOpt
from layers.dizzyRNNCellOptBottom import DizzyRNNCellOptBottom
from layers.dizzyRNNCellv1 import DizzyRNNCellV1
from layers.dizzyRNNCellv2 import DizzyRNNCellV2

def buildRNNCells(layer_type, state_size, num_stacked):
    if layer_type == 1:
        rnn_cell = tf.nn.rnn_cell.LSTMCell(state_size)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * num_stacked)
    elif layer_type == 2:
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * num_stacked)
    elif layer_type == 3:
        rnn_cell = DizzyRNNCellV1(state_size)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
            [[rnn_cell] * num_stacked])
    elif layer_type == 4:
        rnn_cell = DizzyRNNCellV2(state_size)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
            [[rnn_cell] * num_stacked])
    elif layer_type == 5:
        rnn_cell = tf.nn.rnn_cell.GRUCell(state_size)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * num_stacked)
    elif layer_type == 6:
        rnn_cell = DizzyRNNCellOpt(state_size)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
            [DizzyRNNCellOptBottom(state_size)] + [rnn_cell] * (num_stacked-1))

    return stacked_cell
