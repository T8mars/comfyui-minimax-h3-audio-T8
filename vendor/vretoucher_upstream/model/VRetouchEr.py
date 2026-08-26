''' Towards An End-to-End Framework for Video Inpainting
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.modules.spectral_norm import spectral_norm as _spectral_norm
from model.network_vrt_pair_qkv_video_fuse import Stage
from model.gpen_model_video import Decoder_noise, Encoder_new
from model.modules.flow_comp import SPyNet, flow_warp

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).' %
            (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0,
                 scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=1, padding=padding)
        self.scale_factor = scale_factor
    def forward(self, x):
        x = self.conv(x)
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True, recompute_scale_factor=False)

class Fuse(nn.Module):
    def __init__(self,
                 input_channel=512,
                 output_channel=512):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, 1), nn.LeakyReLU())
        self.conv_2 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, 1), nn.LeakyReLU())
        self.conv_3 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, 1), nn.LeakyReLU())
    def forward(self, x, alpha):
        x = self.conv_1(x)
        alpha = self.conv_2(alpha)
        return self.conv_3(x + alpha)

class InpaintGenerator(BaseNetwork):
    def __init__(self, n_layer_t=5, frame_num=6):
        super(InpaintGenerator, self).__init__()
        # encoder
        self.encoder = Encoder_new(
            size = 512,
            channel_multiplier=2,
            narrow=1,
            device='cuda')

        # decoder
        self.decoder = Decoder_noise(
            size = 512,
            style_dim = 8,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            isconcat=True,
            narrow=1,
            device='cuda'
        )
        self.vrt = Stage(
                 n_layer_t = n_layer_t,
                 in_dim = 512,
                 dim = 512,
                 input_resolution = (6, 64, 64),
                 depth = 5,
                 num_heads = 8,
                 window_size = [6, 16, 16],
                 mul_attn_ratio=0.9,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False)
        self.norm = nn.LayerNorm(512)
        self.frame_num = frame_num
        self.BlemishLSTM = BlemishLSTM(frames = frame_num)
        self.conv_512 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.LeakyReLU()
        )
        self.conv_256 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.LeakyReLU()
        )
        self.conv_128 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1), nn.LeakyReLU()
        )
        self.back_512 = nn.Sequential(
            deconv(512, 256, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
            deconv(256, 128, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
            deconv(128, 64, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
        )
        self.back_256 = nn.Sequential(
            deconv(512, 256, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
            deconv(256, 128, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU()
        )
        self.back_128 = nn.Sequential(
            deconv(512, 256, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU()
        )
        self.back_65 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU()
        )
        self.back_64 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU()
        )
        self.back_63 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU()
        )

        self.update_spynet = SPyNet()
        self.conv = nn.Sequential(deconv(512, 256, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
                                  deconv(256, 128, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
                                  nn.Conv2d(128, 64, 3, 1, 1), nn.LeakyReLU(),
                                  nn.Conv2d(64, 1, 3, 1, 1), nn.LeakyReLU())

    def comput_flow(self, masked_local_frames):
        b, l_t, c, h, w = masked_local_frames.size()
        # compute forward and backward flows of masked frames
        masked_local_frames = F.interpolate(masked_local_frames.view(-1, c, h, w), scale_factor=1 / 8, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        masked_local_frames = masked_local_frames.view(b, l_t, c, h // 8, w // 8)
        mlf_1 = masked_local_frames[:, :-1, :, :, :].reshape(-1, c, h // 8, w // 8)
        mlf_2 = masked_local_frames[:, 1:, :, :, :].reshape(-1, c, h // 8, w // 8)
        pred_flows_forward = self.update_spynet(mlf_1, mlf_2)
        pred_flows_forward = pred_flows_forward.view(l_t-1, b, h // 8, w // 8, 2)
        return pred_flows_forward

    def forward(self, source_tensor_list):
        source_tensor_list = torch.stack(source_tensor_list, dim=1)
        B, T, C, H, W = source_tensor_list.shape
        flow_list = self.comput_flow(source_tensor_list)
        source_tensor_list = source_tensor_list.reshape(T, B, C, H, W)
        decoder_noise, x_list = self.encoder(source_tensor_list.clone()) # decoder_noise是关键帧中每一层的信息；x_list是所有的帧
        vrt = torch.stack([self.conv_512(decoder_noise[0]),
                           self.conv_256(decoder_noise[1]),
                           self.conv_128(decoder_noise[2]), decoder_noise[3], decoder_noise[4], decoder_noise[5]], dim=2)
        x_list = torch.stack(x_list, dim=0)
        B, C, T, H, W = vrt.shape  # x_list是所有帧，对它中的信息进行对齐
        attention_feat = self.BlemishLSTM(x_list, flow_list, source_tensor_list, self.frame_num)
        mask_list = []
        vrt = self.vrt(vrt, x_list.reshape(B, C, T, H, W), attention_feat[:, :, (self.frame_num-T):, :, :]).reshape(B, T, H, W, C)
        vrt = self.norm(vrt).reshape(T, B, C, H, W)
        attention_feat = attention_feat.reshape(self.frame_num, B, C, H, W)
        feature = [512, 256, 128, 65, 64, 63]
        for i in range(self.frame_num):
            mask_list.append(self.conv(attention_feat[i]))
            if i < 6:
                block = getattr(self, f"back_{feature[i]}")
                decoder_noise[i] = decoder_noise[i] + block(vrt[i])
        result = self.decoder(decoder_noise[::-1])
        return result, mask_list, flow_list

# ######################################################################
#  Discriminator for Temporal Patch GAN
# ######################################################################

class img_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_img = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(64, 256, 3, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1), nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0))
        self.conv_feat = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.LeakyReLU())
    def forward(self, x, img):
        sigmoid_params = self.conv_img(img) # torch.Size([1, 2, res, res])
        x = self.conv_feat(x)
        alpha, beta = torch.split(sigmoid_params, 512, dim=1)
        return torch.sigmoid(x*alpha + beta)


class ConvLSTMCell(nn.Module):
    def __init__(self):
        super(ConvLSTMCell, self).__init__()
        self.dot = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1), nn.LeakyReLU())
        self.add = nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride=1, padding=1), nn.LeakyReLU())
        self.init = nn.Conv2d(512, 1024, 1, stride=1, padding=0)
        # self.conv = nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv_1 = Fuse()
        self.conv_2 = Fuse()

    def init_hidden(self, blemish):
        dot, add = torch.split(self.init(blemish), 512, dim=1)
        return dot, add

    def forward(self, blemish, blemish_flow, state):
        dot = self.dot(torch.cat([blemish_flow, state[0]], dim=1))
        add = self.add(torch.cat([blemish_flow, state[1]], dim=1))
        blemish_flow = self.conv_1((blemish_flow * dot + add), blemish)
        blemish = torch.sigmoid(self.conv_2(blemish, blemish_flow))
        return blemish, dot, add



class BlemishLSTM(nn.Module):
    def __init__(self, hidden_channels=512, frames=6, num_layers=3):
        super(BlemishLSTM, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.frames = frames
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell()
            setattr(self, name, cell)
            self._all_layers.append(cell)
        self.img_attention = img_attention()


    def forward(self, x_list, flow_list, img_list, T):
        blemish_list = []
        internal_state = []
        blemish = self.img_attention(x_list[0], img_list[0])
        blemish_list.append(blemish)
        # initialize
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            bsize, _, height, width = blemish.size()
            (dot, add) = getattr(self, name).init_hidden(blemish)
            internal_state.append((dot, add))
        for step in range(1, T):
            blemish = self.img_attention(x_list[-1], img_list[step])
            blemish_flow = flow_warp(blemish_list[step - 1], flow_list[step - 1])
            for i in range(self.num_layers):
                name = 'cell{}'.format(i)
                blemish, dot, add = getattr(self, name)(blemish, blemish_flow, internal_state[i])
                internal_state[i] = (dot, add)
            blemish_list.append(blemish)
        out = torch.stack(blemish_list, dim=2)
        return out

