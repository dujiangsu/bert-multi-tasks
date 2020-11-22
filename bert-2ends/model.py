import os
import sys
import logging
import torch
import time
from torch import nn
from downstream import SequenceClassification
from data import GlueDataArgs, DataIterator
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import glue_tasks_num_labels

class MultiTaskModel: