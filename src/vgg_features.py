import torch.nn as nn
import torchvision.models as models
import torch

class VGGFeatureExtractor(nn.Module):
    '''
    En conjunto, este código prepara un módulo que toma una imagen de entrada, 
    la normaliza de ser necesario y la pasa por la VGG19, devolviendo los mapas 
    de características en los índices indicados por layer_ids.
    
    Esos mapas se usarán para calcular la pérdida perceptual comparando 
    características de la imagen generada con las de la imagen real.
    '''
    def __init__(self, layer_ids=[35], use_input_norm=True, device="cpu"):
        """
        layer_ids: índices de capas cuyas salidas se usarán, limita para que no haga vaya mas halla de lo necesario
        use_input_norm: normaliza la entrada con los valores de ImageNet.
        """
        super(VGGFeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features.to(device).eval()
        
        self.use_input_norm = use_input_norm
        if use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) #Estos son valores estandar en python para normalizar 
            self.register_buffer('std',  torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
        # Construir un módulo que incluya todas las capas hasta la máxima layer_id
        max_id = max(layer_ids)
        self.features = nn.Sequential(*[vgg19[i] for i in range(max_id+1)])
        self.layer_ids = layer_ids

    
    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        outputs = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.layer_ids:
                outputs.append(x)
        return outputs