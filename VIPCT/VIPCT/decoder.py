# This file contains the code for the decoder in VIP-CT framework.
# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper described in the readme file:
# Roi Ronen, Vadim Holodovsky and Yoav. Y. Schechner, "Variable Imaging Projection Cloud Scattering Tomography",
# Proc. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.
#
# Copyright (c) Roi Ronen. The python code is available for
# non-commercial use and exploration.  For commercial use contact the
# author. The author is not liable for any damages or loss that might be
# caused by use or connection to this code.
# All rights reserved.
#
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from .mlp_function import MLPWithInputSkips2


def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)

class Decoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
            self,
            type,
            average_cams,
            feature_flatten,
            latent_size,
            use_neighbours,
            device,
            multiply_factors = [1, 1, 1],
    ):
        """
        :param type Decoder network type.
        """
        super().__init__()
        self.average_cams = average_cams
        self.feature_flatten = feature_flatten
        out_size = 27 if use_neighbours else 1
        self.type = type
        self.device = device
        if type == 'FixCT':
            linear1 = torch.nn.Linear(latent_size, 2048)
            linear2 = torch.nn.Linear(2048, 512)
            linear3 = torch.nn.Linear(512, 64)
            linear4 = torch.nn.Linear(64, out_size)
            _xavier_init(linear1)
            _xavier_init(linear2)
            _xavier_init(linear3)
            _xavier_init(linear4)

            self.decoder = torch.nn.Sequential(
                linear1,
                # torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(True),
                linear2,
                # torch.nn.BatchNorm1d(64),
                torch.nn.ReLU(True),
                linear3,
                torch.nn.ReLU(True),
                linear4
            )
        elif type == 'FixCTv2':
            linear1 = torch.nn.Linear(latent_size, 2048)
            linear2 = torch.nn.Linear(2048, 512)
            linear3 = torch.nn.Linear(512, 64)
            linear4 = torch.nn.Linear(64, out_size)
            _xavier_init(linear1)
            _xavier_init(linear2)
            _xavier_init(linear3)
            _xavier_init(linear4)

            self.decoder = torch.nn.Sequential(
                linear1,
                torch.nn.LayerNorm(2048),
                torch.nn.ReLU(True),
                linear2,
                torch.nn.LayerNorm(512),
                torch.nn.ReLU(True),
                linear3,
                torch.nn.LayerNorm(64),
                torch.nn.ReLU(True),
                linear4
            )
        elif type == 'FixCTv3':
            linear1 = torch.nn.Linear(latent_size, 2048)
            linear2 = torch.nn.Linear(2048, 512)
            linear3 = torch.nn.Linear(512, 64)
            linear4 = torch.nn.Linear(64, out_size)
            _xavier_init(linear1)
            _xavier_init(linear2)
            _xavier_init(linear3)
            _xavier_init(linear4)
            nn.init.constant_(linear4.bias.data, 10)

            self.decoder = torch.nn.Sequential(
                linear1,
                torch.nn.LayerNorm(2048),
                torch.nn.ReLU(True),
                linear2,
                torch.nn.LayerNorm(512),
                torch.nn.ReLU(True),
                linear3,
                torch.nn.LayerNorm(64),
                torch.nn.ReLU(True),
                linear4
            )
        elif type == 'FixCTv4':

            self.decoder = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),

            ),
            torch.nn.Linear(512, out_size))

        elif type == 'FixCTv4_microphysics':

            self.decoder = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),

            ),
            torch.nn.Linear(512, 3*out_size),)
            #torch.nn.Sigmoid(),
            #Multiply(multiply_factors,device))

        elif type == 'FixCTv4_microphysics_4out':

            self.decoder = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),

            ),
            torch.nn.Linear(512, 4*out_size),)
            #torch.nn.Sigmoid(),
            #Multiply(multiply_factors,device))

        elif type == 'FixCTv4_microphysics_seperated':

            self.decoder0 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),
                ),
                torch.nn.Linear(512, out_size),)
            self.decoder1 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                    8,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    512,
                    input_skips=(5,),
                ),
                torch.nn.Linear(512, out_size),)
            self.decoder2 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                    8,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    512,
                    input_skips=(5,),
                ),
                torch.nn.Linear(512, out_size),)

        elif type == 'FixCTv4_microphysics_seperated_w_score':

            self.decoder0 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),
                ),
                torch.nn.Linear(512, 2*out_size),)
            self.decoder1 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                    8,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    512,
                    input_skips=(5,),
                ),
                torch.nn.Linear(512, out_size),)
            self.decoder2 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                    8,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    512,
                    input_skips=(5,),
                ),
                torch.nn.Linear(512, out_size),)

        elif type == 'FixCTv4_microphysics_seperated_w_sigmoid':

            self.decoder0 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),
                ),
                torch.nn.Linear(512, out_size),
                torch.nn.Sigmoid(),
                Multiply(multiply_factors[0],device))
            self.decoder1 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                    8,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    512,
                    input_skips=(5,),
                ),
                torch.nn.Linear(512, out_size),
                torch.nn.Sigmoid(),
                Multiply(multiply_factors[1], device))
            self.decoder2 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                    8,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    512,
                    input_skips=(5,),
                ),
                torch.nn.Linear(512, out_size),
                torch.nn.Sigmoid(),
                Multiply(multiply_factors[2], device))

        elif type == 'FixCTv4_1micro':

            self.decoder = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),
            ),
            torch.nn.Linear(512, out_size),
            torch.nn.Sigmoid(),
            Multiply(multiply_factors, device))

        elif type == 'FixCTv4_1micro_2':

            self.decoder = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),
            ),
            torch.nn.Linear(512, out_size),
            torch.nn.Softplus())

        elif type == 'FixCTv4_mask':

            self.decoder = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),
            ),
            torch.nn.Linear(512, out_size),
            torch.nn.Sigmoid(),)


    def forward(self, x):
        if self.average_cams:
            x = torch.mean(x,1)
        if self.feature_flatten:
            x = x.reshape(x.shape[0],-1)
        if 'score' in self.type:
            dec0_out = self.decoder0(x)
            return torch.stack([dec0_out[:,0], torch.sigmoid(dec0_out[:,1]), self.decoder1(x).squeeze(), self.decoder2(x).squeeze()], dim=-1)
        elif 'seperated' in self.type:
            return torch.stack([self.decoder0(x),self.decoder1(x),self.decoder2(x)],dim=-1).squeeze()
        elif '4out' in self.type:
            dec_out = self.decoder(x)
            return torch.stack([dec_out[:, 0], torch.sigmoid(dec_out[:, 1]), dec_out[:, 2], dec_out[:, 3]],dim=-1)
        else:
            return self.decoder(x)

    @classmethod
    def from_cfg(cls, cfg, latent_size, use_neighbours=False):
        if hasattr(cfg.decoder, 'out_factors'):
            out_factors = cfg.decoder.out_factors
        else:
            out_factors = None
        return cls(
            type = cfg.decoder.name,
            average_cams=cfg.decoder.average_cams,
            feature_flatten=cfg.decoder.feature_flatten,
            latent_size = latent_size,
            use_neighbours = use_neighbours,
            device=cfg.gpu,
            multiply_factors = out_factors,
        )

class Multiply(nn.Module):
  def __init__(self,multiply_factors, device):
    super(Multiply, self).__init__()
    self.multiply_factors = torch.tensor(multiply_factors,device=device)
  def forward(self, tensors):
    result = self.multiply_factors*tensors
    return result