import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import os

import torch.nn.functional as F

class TestyModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.

        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_cyc * ||G_B(G_A(A)) - A||
        Backward cycle loss: lambda_cyc * ||G_A(G_B(B)) - B||
        Identity loss (optional): lambda_idt * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A)
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'G_CLS', 'D_CLS', 'color_A']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'temp_B', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_temp_B', 'rec_B']
        

        AtoB = self.opt.direction == 'AtoB'
        if AtoB: 
            self.visual_names = visual_names_A
        else:
            self.visual_names = visual_names_B


        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG_A, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG_B, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.alias_net = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, "antialias", opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # Load parameters
        print('--------Load AliasNet--------')
        load_path = './alias_net.pth'
        state_dict = torch.load(load_path)
        for p in list(state_dict.keys()):
            #why delete module. params???
            state_dict["module."+str(p)] = state_dict.pop(p)
        self.alias_net.load_state_dict(state_dict)
        for p in self.alias_net.parameters():
            p.requires_grad = False

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.B_gray = input['B_gray'].to(self.device)
        self.label = input['label'].to(self.device)  # [4]
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):

        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        self.temp_B = self.netG_A(self.real_A, self.B_gray)
        # visualizing the cell size code
        #cellcode = F.normalize(cellcode, p=2,dim=1)
        #cellcode = torch.squeeze(cellcode,0)
        #print(cellcode.shape)
        self.fake_B = self.alias_net(self.temp_B)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_temp_B = self.netG_A(self.fake_A, self.B_gray)# cellcode#
        self.rec_B = self.alias_net(self.rec_temp_B)   # G_A(G_B(B))

            
    def RGB2GRAY(self, RGB_image):
            gray_image = None
            for i in range(RGB_image.shape[0]):
                R = RGB_image[i][0]
                G = RGB_image[i][1]
                B = RGB_image[i][2]
                gray = 0.299 * R + 0.587 * G + 0.114 * B  # [256,256]
                gray = torch.unsqueeze(gray, 0)  # [1,256,256]
                gray = torch.cat([gray, gray, gray], 0)  # [3,256,256]
                if i == 0:
                    gray_image = torch.unsqueeze(gray, 0)
                else:
                    gray = torch.unsqueeze(gray, 0)
                    gray_image = torch.cat([gray_image, gray], 0)
            #print(gray_image.shape)
            return gray_image


    def optimize_parameters(self):
        """No optimization for test model."""
        pass

