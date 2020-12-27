import numpy as np
from PIL import Image
from utils import FDA_source_to_target_np
import scipy.misc

im_src = Image.open("../../scratch/data/cityscapes/leftImg8bit/train/tubingen/tubingen_000015_000019_leftImg8bit.png").convert('RGB')
im_trg = Image.open("../../scratch/data/dark_zurich/val/rgb_anon/val/night/GOPR0356/GOPR0356_frame_000330_rgb_anon.png").convert('RGB')

im_src = im_src.resize( (1024,512), Image.BICUBIC )
im_trg = im_trg.resize( (1024,512), Image.BICUBIC )

im_src = np.asarray(im_src, np.float32)
im_trg = np.asarray(im_trg, np.float32)

im_src = im_src.transpose((2, 0, 1))
im_trg = im_trg.transpose((2, 0, 1))

src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.01 )

src_in_trg = src_in_trg.transpose((1,2,0))
scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save('demo_images/src1_in_trg3.png')

