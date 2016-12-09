import tensorflow as tf
import numpy as np
from layers.dizzyRNNCellOpt import DizzyRNNCellOpt
from layers.dizzyRNNCellv1 import DizzyRNNCellV1
from layers.dizzyRNNCellv2 import DizzyRNNCellV2
from layers.iRNNCell import IRNNCell
from layers.decompRNNCell import DecompRNNCell
from layers.dizzyRNNCell import DizzyRNNCell

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
            [rnn_cell] * num_stacked)
    elif layer_type == 4:
        rnn_cell = DizzyRNNCellV2(state_size)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
            [rnn_cell] * num_stacked)
    elif layer_type == 5:
        rnn_cell = tf.nn.rnn_cell.GRUCell(state_size)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * num_stacked)
    elif layer_type == 6:
        bottom_cell = DizzyRNNCellOpt(state_size, bottom=True)
        rnn_cell = DizzyRNNCellOpt(state_size, bottom=False)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
            [bottom_cell] + [rnn_cell] * (num_stacked-1))
    elif layer_type == 7:
        bottom_cell = IRNNCell(state_size, bottom=True)
        rnn_cell = IRNNCell(state_size, bottom=False)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
            [bottom_cell] + [rnn_cell] * (num_stacked-1))
    elif layer_type == 8:
        stacked_cell = DizzyRNNCell(state_size, bottom=True)
        #  bottom_cell = DizzyRNNCell(state_size, bottom=True)
        #  rnn_cell = DizzyRNNCell(state_size, bottom=False)
        #  stacked_cell = tf.nn.rnn_cell.MultiRNNCell(



    return stacked_cell
