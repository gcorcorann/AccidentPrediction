#!/usr/bin/env python3
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

class SingleFrame(nn.Module):
    """Single Frame Accident Classifier."""

    def __init__(self):
        super().__init__()
        self.cnn = models.vgg11(pretrained=False)
        num_fts = self.cnn.classifier[3].in_features
        self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-4]
                )
        self.fc = nn.Linear(num_fts, 2)

    def forward(self, inp):
        out = self.cnn.forward(inp)
        print('out:', out.shape)
        out = self.fc(out)
        return out

def main():
    """Main Function."""
    vid_path1 = 'data/000543.mp4'
    vid_path2 = 'data/000589.mp4'
    vid_path3 = 'data/000602.mp4'

    # loss function
    criterion = nn.CrossEntropyLoss()
    # create network object
    net = SingleFrame()

    # store video data
    X1 = []
    X2 = []
    X3 = []
    # read videos
    cap1 = cv2.VideoCapture(vid_path1)
    cap2 = cv2.VideoCapture(vid_path2)
    cap3 = cv2.VideoCapture(vid_path3)
    # for each frame in video
    for i in range(100):
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        _, frame3 = cap3.read()
        # resize frames
        frame1 = cv2.resize(frame1, (224,224))
        frame2 = cv2.resize(frame2, (224,224))
        frame3 = cv2.resize(frame3, (224,224))
        # convert to RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
        # store in lists
        X1.append(frame1)
        X2.append(frame2)
        X3.append(frame3)

    # convert lists to ndarray
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    # reformat into [sequenceLen x batchSize x numChannels x Width x Height]
    X = np.stack((X1, X2, X3))
    X = X.transpose(1, 0, 4, 2, 3)
    # create labels
    y = 90
    labels = torch.zeros(100).type(torch.LongTensor)
    labels[90:] = 1
    # wrap in Variable
    inputs = Variable(torch.from_numpy(X)).type(torch.FloatTensor)
    labels = Variable(labels)
    print('inputs:', inputs.shape)
    print('labels:', labels.shape)

    # for each batch in sequence
    for i, inp in enumerate(inputs):
        print(i)
        print('inp:', inp.shape)
        # pass through network
        output = net.forward(inp)
        print('output:', output.data)
        loss = criterion(output,
#        log_loss = torch.sum(torch.log(output[:, 1]))
#        m = math.exp(-max(0, y - i))
#        print('m:', m)
#        print('log_loss:', log_loss.data[0])
#        loss = -m * log_loss
#        print('loss:', loss.data[0])
#        print()

if __name__ == '__main__':
    main()
