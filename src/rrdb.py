import torch.nn as nn
import torch

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        '''
        Realiza 5 convoluciones en cadena, todos con un kernel de 3x3, stride de 1, y padding de 1.
        Parametros:
            grow_channels: cuantos filtros produce cada convolucion
            in_channels: numero de canales por entrada
        '''
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_channels, growth_channels, 3, 1, 1) #recibe concatenacion de entrada original + salida de conv1
        self.conv3 = nn.Conv2d(in_channels + 2*growth_channels, growth_channels, 3, 1, 1) #recibe entrada original + salida de conv2, etc.
        self.conv4 = nn.Conv2d(in_channels + 3*growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4*growth_channels, in_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True) #Funcion de activacion
        self.scale = 0.2 #Estabilizar el entrenamiento, para que la contribucion sea proporcional a la escala

    def forward(self, x):
        '''
        Realiza las convoluciones y les aplica la funcion de acticacion
        '''
        inputs = x
        x1 = self.lrelu(self.conv1(inputs))
        x2 = self.lrelu(self.conv2(torch.cat([inputs, x1], dim=1))) #FUncion cat concatena inputs con x1, analogo para las lineas de abajo
        x3 = self.lrelu(self.conv3(torch.cat([inputs, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([inputs, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([inputs, x1, x2, x3, x4], dim=1))
        return inputs + x5 * self.scale

class RRDB(nn.Module):
    '''
    Agrupa 3 ResidualDenseBlock en serie.
    '''
    def __init__(self, in_channels, growth_channels=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(in_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(in_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(in_channels, growth_channels)
        self.scale = 0.2

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.scale