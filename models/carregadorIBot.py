# carregador do ibot oficial
# modelo: https://github.com/bytedance/ibot

import os
import sys
import torch
import torch.nn as nn
from functools import partial

# caminho raiz do projeto
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, 'ibot_repo'))


# essa funcao e auxiliar para drop path
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


# classe DropPath do transformer
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# classe MLP do transformer
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# classe de atencao do transformer
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# classe bloco do transformer (atencao + mlp)
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        y = self.attn(self.norm1(x))
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# classe que divide imagem em patches
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# classe principal do vision transformer do ibot
class VisionTransformerIBot(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                       in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], 
                  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # inicializacao dos pesos
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self.initWeights)

    def initWeights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepareTokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        clsTokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((clsTokens, x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepareTokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


# essa funcao cria o vit small (384 dim, 6 heads)
def vitSmall(patchSize=16, **kwargs):
    model = VisionTransformerIBot(
        patch_size=patchSize, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model


# essa funcao cria o vit base (768 dim, 12 heads)
def vitBase(patchSize=16, **kwargs):
    model = VisionTransformerIBot(
        patch_size=patchSize, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model


# essa classe extrai features do ibot usando os 12 blocos do ViT
class ExtratorIBot:
    
    def __init__(self, modelo='vit_small', dispositivo='auto'):
        if dispositivo == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = dispositivo
        
        self.tipoModelo = modelo
        self.numBlocos = 12
        
        # dimensoes por modelo
        if modelo == 'vit_small':
            self.dimEmbed = 384
        else:
            self.dimEmbed = 768
        
        # todos os 12 blocos
        self.dimBlocos = {f'block{i}': self.dimEmbed for i in range(12)}
        
        self.features = {}
        self.hooks = []
        
        self.carregarModelo()
    
    # essa funcao carrega o modelo e os pesos
    def carregarModelo(self):
        
        print(f"[IBot] Carregando modelo {self.tipoModelo}...")
        
        if self.tipoModelo == 'vit_small':
            self.model = vitSmall()
            caminhoPesos = os.path.join(RAIZ, 'checkpoint_ibot_vits16.pth')
        else:
            self.model = vitBase()
            caminhoPesos = os.path.join(RAIZ, 'checkpoint_ibot_vitb16.pth')
        
        # carrega pesos
        if os.path.exists(caminhoPesos):
            checkpoint = torch.load(caminhoPesos, map_location='cpu', weights_only=False)
            
            if 'state_dict' in checkpoint:
                pesos = checkpoint['state_dict']
            elif 'model' in checkpoint:
                pesos = checkpoint['model']
            else:
                pesos = checkpoint
            
            # remove prefixos
            pesosLimpos = {}
            for chave, valor in pesos.items():
                novaChave = chave.replace('backbone.', '').replace('module.', '')
                pesosLimpos[novaChave] = valor
            
            self.model.load_state_dict(pesosLimpos, strict=False)
            print(f"[IBot] Pesos carregados!")
        else:
            print(f"[IBot] ERRO: pesos nao encontrados")
            raise FileNotFoundError(f"Arquivo nao encontrado: {caminhoPesos}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.registrarHooks()
        
        print(f"[IBot] Repositorio: https://github.com/bytedance/ibot")
        print(f"[IBot] Pre-treino: Self-supervised (Masked Image Modeling) no ImageNet-1K")
        print(f"[IBot] Dispositivo: {self.device}")
        print(f"[IBot] Blocos: 12")
        print(f"[IBot] Dimensao: {self.dimEmbed}")
    
    # essa funcao registra hooks nos blocos para capturar features
    def registrarHooks(self):
        
        def criarHook(indice):
            def funcaoHook(modulo, entrada, saida):
                self.features[f'block_{indice}'] = saida
            return funcaoHook
        
        for h in self.hooks:
            h.remove()
        self.hooks = []
        
        for i, bloco in enumerate(self.model.blocks):
            if i < self.numBlocos:
                h = bloco.register_forward_hook(criarHook(i))
                self.hooks.append(h)
        
        print(f"[IBot] Registrados {len(self.hooks)} hooks")
    
    # essa funcao extrai features de todos os 12 blocos usando media dos patches
    def extrairFeatures(self, x, aplicarGAP=True):
        
        self.features = {}
        
        with torch.no_grad():
            _ = self.model(x)
        
        resultado = {}
        
        for i in range(self.numBlocos):
            chave = f'block_{i}'
            
            if chave in self.features:
                feat = self.features[chave]
                
                if aplicarGAP:
                    if len(feat.shape) == 3:
                        # media dos patches (ignora CLS)
                        feat = feat[:, 1:].mean(dim=1)
                    elif len(feat.shape) == 4:
                        feat = feat.mean(dim=[2, 3])
                
                resultado[f'block{i}'] = feat.cpu()
        
        return resultado
    
    # essa funcao extrai features usando CLS token de todos os 12 blocos
    def extrairFeaturesComCLS(self, x):
        
        self.features = {}
        
        with torch.no_grad():
            _ = self.model(x)
        
        resultado = {}
        
        for i in range(self.numBlocos):
            chave = f'block_{i}'
            
            if chave in self.features:
                feat = self.features[chave]
                if len(feat.shape) == 3:
                    feat = feat[:, 0]
                resultado[f'block{i}'] = feat.cpu()
        
        return resultado
    
    def pegarDimensoes(self):
        return self.dimBlocos.copy()
    
    @property
    def stageDims(self):
        return self.dimBlocos


# essa funcao cria o extrator de features do ibot
def criarExtrator(modelo='vit_small', dispositivo='auto'):
    return ExtratorIBot(modelo=modelo, dispositivo=dispositivo)
