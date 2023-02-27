import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import os
import torch.nn.functional as F
from .patchnce import PatchNCELoss


class NCEModel(BaseModel):
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
        if is_train:
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
            parser.add_argument('--lambda_LMC', type=float, default=1.0, help='')
            parser.add_argument('--lambda_NCE_X', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
            parser.add_argument('--lambda_NCE_Y', type=float, default=1.0, help='weight for NCE loss: NCE(G(Y), Y)')
            parser.add_argument('--nce_layers', type=str, default='0,1,2,3,6', help='compute NCE loss on which layers')
            
            parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
            parser.add_argument('--netF_nc', type=int, default=256)
            parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
            parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
            # parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            # parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=10.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt=opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'G_CLS', 'D_CLS', 'color_A']
        self.loss_names = ['D_A', 'G_A', 'NCE_X', 'NCE_Y', 'G_CLS', 'D_CLS', 'color_A']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # visual_names_A = ['real_A', 'temp_B', 'fake_B', 'rec_A']
        # visual_names_B = ['real_B', 'fake_A', 'rec_temp_B', 'rec_B']
        visual_names_A = ['real_A', 'fake_B','real_B','idt_B']
        # if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            # visual_names_A.append('idt_B')
            # visual_names_B.append('idt_A')


        # self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        self.visual_names = visual_names_A
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if opt.isTrain:
            # self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            self.model_names=['G_A','D_A','F']
        else:  # during test time, only load Gs
            # self.model_names = ['G_A', 'G_B']
            self.model_names = ['G_A']


        # define networks (both Generators and discriminators)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG_A, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,pretrained=opt.pretrained,is_train=opt.isTrain)
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG_B, opt.norm,
                                        # not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.alias_net = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, "antialias", opt.norm,
                                        #    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netF=networks.define_F(opt.input_nc, opt.netF, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)



        if opt.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, 'CPDis_cls',
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            # opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if opt.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            # self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images


            ## define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            # self.criterionCycle = torch.nn.L1Loss()
            # self.criterionIdt = torch.nn.L1Loss()
            self.criterionCrossEntropy = torch.nn.CrossEntropyLoss().to(self.device)
            # self.criterionL2 = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss() #@pw: no need to .to(self.device)?
            # self.blur_rgb = networks.Blur(3).to(self.device)
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))


            ## initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(self.netD_B.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_cls = torch.optim.Adam(self.netD_A.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_F=torch.optim.Adam(self.netF.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))

            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D_cls)
            # self.optimizers.append(self.optimizer_F)

        # Load parameters
        # print('--------Load AliasNet--------')
        # load_path = './alias_net.pth'
        # state_dict = torch.load(load_path)
        # for p in list(state_dict.keys()):
        #     state_dict["module."+str(p)] = state_dict.pop(p)
        # self.alias_net.load_state_dict(state_dict)
        # for p in self.alias_net.parameters():
        #     p.requires_grad = False

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

    # def RGB2GRAY(self, RGB_image):
    #     gray_image = None
    #     for i in range(RGB_image.shape[0]):
    #         R = RGB_image[i][0]
    #         G = RGB_image[i][1]
    #         B = RGB_image[i][2]
    #         gray = 0.299 * R + 0.587 * G + 0.114 * B  # [256,256]
    #         gray = torch.unsqueeze(gray, 0)  # [1,256,256]
    #         gray = torch.cat([gray, gray, gray], 0)  # [3,256,256]
    #         if i == 0:
    #             gray_image = torch.unsqueeze(gray, 0)
    #         else:
    #             gray = torch.unsqueeze(gray, 0)
    #             gray_image = torch.cat([gray_image, gray], 0)
    #     #print(gray_image.shape)
    #     return gray_image


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.temp_B = self.netG_A(self.real_A, self.B_gray)
        # visualizing the cell size code
        #cellcode = F.normalize(cellcode, p=2,dim=1)
        #cellcode = torch.squeeze(cellcode,0)
        #print(cellcode.shape)
        # self.fake_B = self.alias_net(self.temp_B)  # G_A(A)
        # self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        # self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        # self.rec_temp_B = self.netG_A(self.fake_A, self.B_gray)# cellcode#
        # self.rec_B = self.alias_net(self.rec_temp_B)   # G_A(G_B(B))

        self.fake_B=self.netG_A(self.real_A,self.B_gray)
        self.idt_B=self.netG_A(self.real_B,self.B_gray)
        
    # def backward_D_basic(self, netD, real, fake):
    #     """Calculate GAN loss for the discriminator

    #     Parameters:
    #         netD (network)      -- the discriminator D
    #         real (tensor array) -- real images
    #         fake (tensor array) -- images generated by a generator

    #     Return the discriminator loss.
    #     We also call loss_D.backward() to calculate the gradients.
    #     """
    #     # Real
    #     pred_real = netD(real)
    #     loss_D_real = self.criterionGAN(pred_real, True)
    #     # Fake
    #     pred_fake = netD(fake.detach())
    #     loss_D_fake = self.criterionGAN(pred_fake, False)
    #     # Combined loss and calculate gradients
    #     loss_D = (loss_D_real + loss_D_fake) * 0.5
    #     loss_D.backward()
    #     return loss_D

    def backward_D_CLS(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        #@pw: CUT uses .mean() after criterionGAN.
        # Real
        pred_real, out_cls = netD(real, self.label)
        loss_D_real = self.criterionGAN(pred_real, True)*self.opt.lambda_GAN
        loss_D_CLS = self.criterionCrossEntropy(out_cls, self.label)*self.opt.lambda_LMC
        # Fake
        pred_fake, out_cls = netD(fake.detach(), self.label)
        loss_D_fake = self.criterionGAN(pred_fake, False)*self.opt.lambda_GAN
        # Combined loss and calculate gradients
        loss_D = ((loss_D_real + loss_D_fake) * 0.5) + loss_D_CLS
        loss_D.backward()
        return ((loss_D_real + loss_D_fake) * 0.5), loss_D_CLS

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        # self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A, self.loss_D_CLS = self.backward_D_CLS(self.netD_A, self.real_B, fake_B)

    # def backward_D_B(self):
    #     """Calculate GAN loss for discriminator D_B"""
    #     fake_A = self.fake_A_pool.query(self.fake_A)
    #     self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self,epoch):
        """Calculate the loss for generators G_A and G_B
        """
        # lambda_idt = self.opt.lambda_identity
        # lambda_A = self.opt.lambda_A
        # lambda_B = self.opt.lambda_B

        # # Identity loss
        # if lambda_idt > 0:
        #     # G_A should be identity if real_B is fed: ||G_A(B) - B||
        #     self.idt_A = self.alias_net(self.netG_A(self.real_B, self.B_gray))
        #     self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) *10
        #     # G_B should be identity if real_A is fed: ||G_B(A) - A||
        #     self.idt_B = self.netG_B(self.real_A)
        #     self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) *10
        # else:
        #     self.loss_idt_A = 0
        #     self.loss_idt_B = 0

        ## GAN loss D_A(G_A(A))
        out_src, out_cls = self.netD_A(self.fake_B,self.label)
        self.loss_G_A = self.criterionGAN(out_src, True)*self.opt.lambda_GAN #@pw:CUT uses .mean() after criterionGAN
        # GAN loss D_B(G_B(B))
        # self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        #print(out_cls,out_cls.shape,self.label.shape,self.label)

        ## LMC loss
        self.loss_G_CLS = self.criterionCrossEntropy(out_cls, self.label)*self.opt.lambda_LMC

        ## l1 loss
        if epoch <=80:
            lambda_l1=8
        else:
            lambda_l1=10
        self.loss_color_A = self.criterionL1(self.fake_B, self.real_A) * lambda_l1
        
        ## cycle loss
        # Forward cycle loss || G_B(G_A(A)) - A||
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * 10
        # Backward cycle loss || G_A(G_B(B)) - B||
        # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * 10

        ## NCE loss
        self.loss_NCE_X=self.calculate_NCE_loss(self.real_A,self.fake_B)*self.opt.lambda_NCE_X
        self.loss_NCE_Y=self.calculate_NCE_loss(self.real_B,self.idt_B)*self.opt.lambda_NCE_Y
        self.loss_NCE_both=(self.loss_NCE_X+self.loss_NCE_Y)*0.5
        #@pw:????虽然判别器算两个gan损失时也乘了0.5

        # # combined loss and calculate gradients
        # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A +\
                    #   self.loss_idt_B + self.loss_color_A + self.loss_G_CLS
        self.loss_G=self.loss_G_A+self.loss_color_A+self.loss_G_CLS+self.loss_NCE_both
        self.loss_G.backward()

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]

        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            # calculate gradients for D
            self.backward_D_A()
            # calculate graidents for G
            self.backward_G(self.opt.epoch_count)
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_F)


    def optimize_parameters(self, i,epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        #@pw:init optimF before model.optimize_parameters()
        # if i==0 and epoch==self.opt.epoch_count:
        #     self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        #     self.optimizers.append(self.optimizer_F)

        # G_A and G_B
        # self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD_A], False)  # Ds require no gradients when optimizing Gs        
        # self.set_requires_grad([self.netF], False)  # Ds require no gradients when optimizing Gs        
        
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.optimizer_F.zero_grad()
        self.backward_G(epoch)             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        self.optimizer_F.step()

        # D_A and D_B
        # self.set_requires_grad([self.netD_B], True)
        # self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        # self.backward_D_B()  # calculate graidents for D_B
        # self.optimizer_D.step()  # update D_A and D_B's weights

        #This condition is nonsence...
        if i % 1 ==0: 
            #print('train!',i)
            self.set_requires_grad([self.netD_A], True)
            self.optimizer_D_cls.zero_grad()
            self.backward_D_A()      # calculate gradients for D_A
            self.optimizer_D_cls.step()

        # self.set_requires_grad([self.netF],True)
        # self.optimizer_F.zero_grad()
        # self.optimizer_F.step()

    
    def calculate_NCE_loss(self,srs,tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_A(tgt, self.B_gray,self.nce_layers, encode_only=True)
        feat_k = self.netG_A(src, self.B_gray,self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, criterion, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = criterion(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers
