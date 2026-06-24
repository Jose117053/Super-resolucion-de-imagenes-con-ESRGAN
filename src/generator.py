from src.rrdb import RRDB
import torch.nn as nn
import torch

class RRDBNet(nn.Module):
    '''
    Red de super‐resolución, Funciona como generador en ESRGAN, aumentando la resolución 
    de la imagen de entrada por un factor scale.

    '''
    def __init__(self, in_channels=3, out_channels=3, num_features=64, growth_channels=32, num_blocks=23, scale=4):
        '''
         Parámetros:
                in_channels : Canales de la imagen de entrada (3 para RGB).
                out_channels : Canales de la imagen de salida (3 para RGB).
                num_features : Número de filtros en la capa inicial y en cada bloque RRDB.
                num_blocks : Número de bloques RRDB que componen la “trunk”. #El trunk es la red que agrupa los bloques RRDB (Un tronco conformado por 23 bloques rrdb)
                scale : Factor de aumento de resolución.
        '''
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        rrdb_blocks = []
        for _ in range(num_blocks):
            rrdb_blocks.append(RRDB(num_features, growth_channels))

            #rrdb_blocks = [RRDB1, RRDB2, RRDB3, ...]
        self.rrdb_trunk = nn.Sequential(*rrdb_blocks) # Equivale a un forward donde el resultado anterior es usado en el actual
                                                        # x = RRDB1.forward(x), x= RRDB2.forward()
        self.trunk_conv = nn.Conv2d(num_features, num_features, 3, 1, 1)
        

        #subidas de resolucion
        upsampling_layers = []
        for _ in range(int(torch.log2(torch.tensor(scale)).item())): #Log2 para ver que tantas iteraciones se hace, si scale igual a 4, entonces log2 de 4 es 2, se harán 2 iteraciones
            upsampling_layers += [
                nn.Conv2d(num_features, num_features * 4, 3, 1, 1), #kernel=3, stride 1, padding 1.
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.upsampling = nn.Sequential(*upsampling_layers) #Lista con todas las capas en orden [conv2d, pixelShuffle, leakyRelu, ...] se repiten los 3 log2 veces

        self.conv_last = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_out = nn.Conv2d(num_features, out_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.rrdb_trunk(fea))
        fea = fea + trunk
        fea = self.upsampling(fea)
        out = self.lrelu(self.conv_last(fea))
        out = self.conv_out(out)
        return out
    
