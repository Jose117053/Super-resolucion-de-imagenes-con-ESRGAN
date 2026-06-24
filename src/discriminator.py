import torch.nn as nn

class Discriminator(nn.Module):
    '''
    Esta clase implementa un discriminador que analiza parches de la imagen de alta resolución
    y produce un mapa de salida indicando la probabilidad de “realismo” para cada
    región. 

    Supongamos que pasamos una imagel real (HR) por el discriminador, obtenemos el siguiente tensor.

    [[ 2.0,   0.5 ],
     [ -1.0,  1.0 ]]

     Es un mapa de 2x2.

     Igualmente al pasar la imagen generada (fake) por el discriminador se produce otro mapa 2x2:

     [[ -0.5,  -1.0 ],
    [  0.2,  -2.0 ]]

    El objetivo del discriminador es convertir los logits en probabilidades y comparar con las etiquetas
    (1 para real, 0 para fake)

    Al aplicar sigmoid al real se devuelve (Esto se calcula en el entrenamiento, pero es para ejemplificar):

    [[ 0.8808,  0.6225 ],
      [ 0.2689,  0.7311 ]]

      Cercanos a 1, son reales, 

      Para la imagen fake:

      [[ 0.3775,  0.2689 ],
      [ 0.5498,  0.1192 ]]

      Hay uno con 54% de ser real, lo cual es falso, pues es la imagen fake


      Nota: aqui no se aplica sigmoide ni el BCEWithLogitLoss, es hasta el entrenamiento.
    '''
    def __init__(self, in_channels=3, base_channels=64):
        '''
        in_channels : Número de canales en la imagen de entrada (normalmente 3 para RGB).
        base_channels: Número de filtros en la primera capa convolucional; las capas 
                        posteriores amplían este número progresivamente.
        '''
        super(Discriminator, self).__init__()

        #Conv 3x3, stide 1, padding 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels), #batch para cada mini batch calcula la media y varainza de las activaaciones y normaliza a media cero.
            nn.LeakyReLU(0.2, inplace=True) #Despues de normalizar leaky relu para considerar tanto positivos como negativos
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv9 = nn.Conv2d(base_channels * 8, 1, 3, 1, 1)

    def forward(self, x):
        x = self.layer1(x)   # [B, 64, H, W]
        x = self.layer2(x)   # [B, 64, H/2, W/2]
        x = self.layer3(x)   # [B, 128, H/2, W/2]
        x = self.layer4(x)   # [B, 128, H/4, W/4]
        x = self.layer5(x)   # [B, 256, H/4, W/4]
        x = self.layer6(x)   # [B, 256, H/8, W/8]
        x = self.layer7(x)   # [B, 512, H/8, W/8]
        x = self.layer8(x)   # [B, 512, H/16, W/16]
        out = self.conv9(x)  # [B, 1, H/16, W/16]
        return out
