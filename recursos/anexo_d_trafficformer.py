import math

import torch
import torch.nn as nn


class TemporalPositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal para secuencias temporales.

    Esta clase implementa una codificación posicional basada en funciones seno y coseno,
    siguiendo la propuesta original del paper "Attention is All You Need". Se aplica directamente
    sobre cada paso temporal de la secuencia de entrada para preservar el orden en modelos
    sin recurrencia.

    La codificación se suma a las features originales antes de ser procesadas por capas densas,
    permitiendo al modelo diferenciar posiciones temporales dentro de la ventana.

    Attributes:
        seq_len (int): Longitud de la secuencia temporal.
        num_features (int): Número de variables por paso temporal.
        pos_encoding (torch.Tensor): Tensor con la codificación posicional precomputada.

    Methods:
        forward(x): Aplica la codificación posicional a un tensor de entrada.
    """
    def __init__(self, seq_len, num_features):
        """
        Inicializa el codificador posicional.

        Args:
            seq_len (int): Longitud de la secuencia temporal (número de pasos).
            num_features (int): Número de variables por paso temporal (dimensión del embedding por paso).
        """
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.pos_encoding = self._build_positional_encoding(seq_len, num_features)  # (seq_len, num_features)

    def _build_positional_encoding(self, seq_len, d_model):
        """
        Genera la matriz de codificación posicional usando funciones seno y coseno.

        Args:
            seq_len (int): Longitud de la secuencia temporal.
            d_model (int): Dimensión del vector de entrada por paso temporal.

        Returns:
            torch.Tensor: Matriz de codificación de tamaño [seq_len, d_model].
        """
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[:(d_model + 1) // 2])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        return pe  # (seq_len, d_model)

    def forward(self, x):
        """
        Suma la codificación posicional al tensor de entrada.

        Args:
            x (torch.Tensor): Tensor de entrada de tamaño [batch_size, num_meters, seq_len, num_features].

        Returns:
            torch.Tensor: Tensor codificado con posición, del mismo tamaño que la entrada.
        """
        pe = self.pos_encoding.to(x.device)  # (seq_len, num_features)
        return x + pe  # broadcasting en eje temporal


class TrafficTemporalFeatureExtractor(nn.Module):
    """
    Extractor de características temporales para secuencias por sensor.

    Esta clase transforma la secuencia temporal de cada sensor (nodo) en un vector de embedding fijo,
    incorporando codificación posicional y un perceptrón multicapa profundo. Está diseñada para
    alimentar un bloque de atención espacial como el utilizado en Trafficformer.

    Arquitectura:
        - Codificación posicional sinusoidal.
        - MLP con las siguientes características:
            - Tres capas lineales con normalización LayerNorm tras cada una.
            - Dropout para regularización y evitar sobreajuste.
            - Activación ReLU para capturar no linealidades.
            - Todos los hiperparámetros principales son configurables.

    Attributes:
        seq_len (int): Longitud de la ventana temporal histórica.
        num_features (int): Número de variables por paso temporal.
        input_dim (int): Dimensión total de entrada (seq_len * num_features).
        pos_encoder (TemporalPositionalEncoding): Módulo de codificación posicional.
        layers (nn.ModuleList): Capas densas de la MLP.
        norms (nn.ModuleList): Normalizaciones LayerNorm.
        dropouts (nn.ModuleList): Capas de regularización por dropout.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        embedding_dim: int,
        hidden_dims=(128, 128),
        dropout=0.2
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.input_dim = seq_len * num_features

        # Codificación posicional temporal
        self.pos_encoder = TemporalPositionalEncoding(seq_len, num_features)

        # Red MLP para mapear a embedding
        dims = [self.input_dim] + list(hidden_dims) + [embedding_dim]
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.norms.append(nn.LayerNorm(dims[i + 1]))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x):
        """
        Aplica el extractor temporal nodo a nodo y devuelve los embeddings temporales.

        Args:
            x (torch.Tensor): Tensor de entrada de shape [batch, num_meters, seq_len, num_features].

        Returns:
            torch.Tensor: Tensor de shape [batch, num_meters, embedding_dim], donde embedding_dim es la dimensión de la representación temporal de cada nodo.
        """
        x = self.pos_encoder(x)  # Codificación posicional
        batch_size, num_meters, T, F = x.shape
        x = x.view(batch_size * num_meters, T * F)

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.norms[i](x)
            x = torch.relu(x)
            if i < len(self.layers) - 1:  # No Dropout after last layer
                x = self.dropouts[i](x)

        x = x.view(batch_size, num_meters, -1)  # [batch, num_meters, embedding_dim]
        return x


class SpatialMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention espacial con soporte para máscaras personalizadas.

    Esta clase adapta el mecanismo estándar de atención multi-cabeza para incorporar
    una máscara espacial arbitraria que restringe la atención solo a los nodos conectados físicamente.
    """

    def __init__(self, embed_dim, num_heads):
        """
        Inicializa el bloque de atención multi-cabeza espacial.

        Args:
            embed_dim (int): Dimensión del embedding de entrada/salida.
            num_heads (int): Número de cabezas de atención.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, spatial_mask=None):
        """
        Aplica atención multi-cabeza sobre los nodos, restringida por la máscara espacial.

        Args:
            x (torch.Tensor): Tensor de shape [batch, num_nodes, embed_dim].
            spatial_mask (torch.Tensor or None): Matriz [num_nodes, num_nodes] booleana o binaria, donde 1 indica conexión.

        Returns:
            torch.Tensor: Tensor de salida tras atención multi-cabeza, de shape [batch, num_nodes, embed_dim].
        """
        # x: [batch, num_nodes, embed_dim]
        # spatial_mask: [num_nodes, num_nodes] -> 1: conectado, 0: no conectado (float/bool)
        attn_mask = None
        if spatial_mask is not None:
            # PyTorch espera mask de shape [num_nodes, num_nodes], True = MASCARA, es decir, IGNORA esos valores.
            attn_mask = ~spatial_mask.bool()  # Invierte la máscara: 1->0 y 0->1

        out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        return out


class TrafficformerEncoderLayer(nn.Module):
    """
    Capa básica de encoder tipo Transformer para interacción espacial.

    Cada capa realiza:
      - Atención multi-cabeza con máscara espacial (residual + normalización).
      - Feedforward no lineal (residual + normalización).
      - Dropout tras cada bloque.
    """

    def __init__(self, embed_dim, num_heads, ff_hidden_dim=None, dropout=0.1):
        """
        Inicializa una capa de encoder Trafficformer.

        Args:
            embed_dim (int): Dimensión del embedding de entrada/salida.
            num_heads (int): Número de cabezas de atención.
            ff_hidden_dim (int, opcional): Tamaño de la capa oculta en el feedforward (por defecto: 2*embed_dim).
            dropout (float): Proporción de dropout tras atención y feedforward.
        """
        super().__init__()
        self.attn = SpatialMultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim or embed_dim * 2),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim or embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, spatial_mask=None):
        """
        Aplica atención multi-cabeza y feedforward con conexiones residuales.

        Args:
            x (torch.Tensor): Tensor de entrada [batch, num_nodes, embed_dim].
            spatial_mask (torch.Tensor or None): Matriz de adyacencia espacial opcional.

        Returns:
            torch.Tensor: Tensor procesado [batch, num_nodes, embed_dim].
        """
        # Multi-head attention (residual + norm)
        attn_out = self.attn(x, spatial_mask=spatial_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feedforward (residual + norm)
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TrafficformerEncoder(nn.Module):
    """
    Módulo encoder compuesto por varias capas Transformer apiladas para modelar la interacción espaciotemporal.

    Este bloque es responsable de propagar y refinar la información espacial entre nodos, permitiendo capturar
    dependencias complejas entre ellos.
    """

    def __init__(self, num_layers, embed_dim, num_heads, ff_hidden_dim=None, dropout=0.1):
        """
        Inicializa el encoder apilando varias capas Transformer.

        Args:
            num_layers (int): Número de capas encoder a apilar.
            embed_dim (int): Dimensión del embedding de entrada/salida.
            num_heads (int): Número de cabezas de atención en cada capa.
            ff_hidden_dim (int, opcional): Tamaño oculto en el bloque feedforward.
            dropout (float): Proporción de dropout.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            TrafficformerEncoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, spatial_mask=None):
        """
        Propaga la información a través de las capas encoder.

        Args:
            x (torch.Tensor): Tensor de entrada [batch, num_nodes, embed_dim].
            spatial_mask (torch.Tensor or None): Matriz de adyacencia espacial opcional.

        Returns:
            torch.Tensor: Tensor de salida tras todas las capas encoder [batch, num_nodes, embed_dim].
        """
        for layer in self.layers:
            x = layer(x, spatial_mask=spatial_mask)
        return x  # [batch, num_nodes, embed_dim]


class SpeedPredictorMLP(nn.Module):
    """
    MLP para predecir el flujo o velocidad final a partir del embedding espaciotemporal de cada nodo.

    Este bloque implementa un perceptrón simple con normalización y activación ReLU.
    """

    def __init__(self, embed_dim, hidden_dim=128):
        """
        Inicializa el predictor final.

        Args:
            embed_dim (int): Dimensión del embedding de entrada.
            hidden_dim (int, opcional): Dimensión de la capa oculta (por defecto 128).
        """
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 1)  # salida escalar por nodo

    def forward(self, x):
        """
        Genera la predicción final para cada nodo.

        Args:
            x (torch.Tensor): Tensor de entrada [batch, num_nodes, embed_dim].

        Returns:
            torch.Tensor: Tensor de salida [batch, num_nodes], predicción por nodo.
        """
        # x: [batch, num_nodes, embed_dim]
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x.squeeze(-1)  # [batch, num_nodes]

class Trafficformer(nn.Module):
    """
    Implementación completa del modelo Trafficformer para predicción de tráfico basada en Transformers.

    Pipeline de procesamiento:
        1. Extracción de embedding temporal por nodo (MLP mejorado).
        2. Interacción espaciotemporal mediante encoder Transformer (máscara espacial opcional).
        3. Predicción de velocidad o flujo por sensor mediante MLP final.

    Este modelo es totalmente configurable en número de capas, dimensiones de embedding y arquitectura
    interna. Puede adaptarse fácilmente a tareas de predicción de tráfico con distintos sensores y ventanas temporales.
    """

    def __init__(self,
        seq_len,
        num_features,
        embedding_dim,
        num_heads,
        num_layers,
        ff_hidden_dim=None,
        dropout=0.1,
    ):
        """
        Inicializa Trafficformer.

        Args:
            seq_len (int): Longitud de la ventana temporal histórica.
            num_features (int): Número de variables por paso temporal.
            embedding_dim (int): Dimensión del embedding intermedio.
            num_heads (int): Número de cabezas de atención en cada capa encoder.
            num_layers (int): Número de capas Transformer en el encoder.
            ff_hidden_dim (int, opcional): Tamaño oculto en la parte feedforward de cada encoder.
            dropout (float): Proporción de dropout.
        """
        super().__init__()
        self.temporal_extractor = TrafficTemporalFeatureExtractor(
            seq_len=seq_len,
            num_features=num_features,
            embedding_dim=embedding_dim,
            hidden_dims=(embedding_dim, embedding_dim),  # ejemplo configurable
            dropout=dropout
        )
        self.encoder = TrafficformerEncoder(
            num_layers=num_layers,
            embed_dim=embedding_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            dropout=dropout
        )
        self.predictor = SpeedPredictorMLP(embed_dim=embedding_dim, hidden_dim=embedding_dim)

    def forward(self, x, spatial_mask=None):
        """
        Realiza la pasada completa del modelo: extracción temporal, interacción espacial y predicción.

        Args:
            x (torch.Tensor): Tensor de entrada [batch, num_nodes, seq_len, num_features].
            spatial_mask (torch.Tensor or None): Matriz de adyacencia espacial opcional [num_nodes, num_nodes].

        Returns:
            torch.Tensor: Tensor de predicción final [batch, num_nodes], estimaciones para cada sensor/nodo.
        """
        # x: [batch, num_nodes, seq_len, num_features]
        temp_embed = self.temporal_extractor(x)          # [batch, num_nodes, embedding_dim]
        st_embed = self.encoder(temp_embed, spatial_mask) # [batch, num_nodes, embedding_dim]
        out = self.predictor(st_embed)                   # [batch, num_nodes]
        return out