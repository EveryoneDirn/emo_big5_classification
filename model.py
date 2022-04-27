import torch
import torch.nn as nn


class EncoderEEGNet(nn.Module):
    def __init__(self, dropout=0.5):
        super(EncoderEEGNet, self).__init__()
        self.conv1=nn.Sequential(
                    nn.Conv2d(in_channels=5, out_channels=16, 
                              kernel_size=(1,5), stride=(1,1), 
                              padding=(0,2), bias=False),
                    nn.BatchNorm2d(16)
                    )
        self.conv2=nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=32, 
                              kernel_size=(2,1), stride=(1,1), 
                              groups=16, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True))
        self.pool1=nn.MaxPool2d(kernel_size=(1,4),
                                stride=(1,4),
                                return_indices=True)
        self.dropout1=nn.Dropout2d(p=dropout)
        ####### separable conv layer ###########
        self.sep_conv = nn.Sequential(
                    nn.Conv2d(in_channels = 32, out_channels = 32, 
                              kernel_size = (1, 15), stride = (1, 1), 
                              padding = (0, 7), bias = False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True))
                    
        self.pool2=nn.MaxPool2d(
                    kernel_size = (1, 4),
                    stride = (1, 4),
                    return_indices=True)
        
        self.dropout2=nn.Dropout(p = dropout)
        
    def forward(self, X):
        feat=self.conv1(X)
        
        feat=self.conv2(feat)
        feat, indicies1=self.pool1(feat)
        feat=self.dropout1(feat)
        
        feat=self.sep_conv(feat)
        feat, indicies2=self.pool2(feat)
        feat=self.dropout2(feat)
        
        return feat, [indicies1, indicies2] 
    
class DecoderEEGNet(nn.Module):
    def __init__(self, dropout=0.5):
        super(DecoderEEGNet, self).__init__()
        
        self.deconv1=nn.Sequential(
                        nn.ConvTranspose2d(in_channels=32, out_channels=32, 
                                           kernel_size = (1, 15), stride = (1, 1), 
                                           padding = (0, 7), bias = False),
                        nn.BatchNorm2d(32),
                        nn.ReLU(True)
                    )
        self.unpool1=nn.MaxUnpool2d(kernel_size=(1,4),
                                    stride=(1,4))
        self.dropout1=nn.Dropout(p=dropout)
        
        self.deconv2=nn.Sequential(
                        nn.ConvTranspose2d(in_channels=32, out_channels=16, 
                                           kernel_size=(2,1), stride=(1,1), 
                                           groups=16, bias=False),
                        nn.BatchNorm2d(16),
                        nn.ReLU(True)
                    )
        self.unpool2=nn.MaxUnpool2d(kernel_size=(1,4),
                                    stride=(1,4))
        self.dropout2=nn.Dropout(p=dropout)
        
        self.deconv3=nn.Sequential(
                        nn.ConvTranspose2d(in_channels=16, out_channels=5, 
                                           kernel_size=(1,5), stride=(1,1), 
                                           padding=(0,2), bias=False),
                        nn.BatchNorm2d(5),
                        nn.Sigmoid()
                    )
                        
                        
    def forward(self, feat, indicies_list):     
        feat=self.unpool1(feat, indicies_list[1])
        feat=self.deconv1(feat)
        feat=self.dropout1(feat)
        feat=self.unpool2(feat, indicies_list[0])
        feat=self.deconv2(feat)
        feat=self.dropout2(feat)
        X_hat=self.deconv3(feat)
        return X_hat
    
    
class AutoEncoderEEGNet(nn.Module):
    def __init__(self, dropout=0.5):
        super(AutoEncoderEEGNet, self).__init__()
        self.encoder=EncoderEEGNet(dropout)
        self.decoder=DecoderEEGNet(dropout)
    
    def forward(self, X):
        code, indicies=self.encoder(X)
        X_hat=self.decoder(code, indicies)
        return X_hat
    
