import argparse
import json
import time

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os
from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr, calc_ssim, AverageMeter


if __name__ == '__main__':
    #start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--ip-dir', type=str, required=True)
    parser.add_argument('--op-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()
    
    ip_path=args.ip_dir
    op_path=args.op_dir
    classnames=os.listdir(ip_path)
    
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    #psnr_dict={}
    #ssim_dict={}
    i=0
    for clas in classnames:
        classpath=os.path.join(ip_path,clas)
        opclasspath=os.path.join(op_path,clas)
        if not os.path.exists(opclasspath):
            os.makedirs(opclasspath)
        filenames=os.listdir(classpath)
        for fil in filenames:
            if not fil.startswith('.'):
                image = pil_image.open(os.path.join(classpath,fil)).convert('RGB')

                orig= np.array(image).astype(np.float32)
                orig=convert_rgb_to_ycbcr(orig)  
                orig = orig[..., 0]
                orig /= 255.
                orig = torch.from_numpy(orig).to(device)
                oig = orig.unsqueeze(0).unsqueeze(0)


                image_width = (image.width // args.scale) * args.scale
                image_height = (image.height // args.scale) * args.scale
                image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
                #image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
                op_name=os.path.join(opclasspath,fil)  
                #image.save(op_name.replace('.', '_downsized_x{}.'.format(args.scale)))
                bi_image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)

                #image.save(op_name.replace('.', '_bicubic_x{}.'.format(args.scale)))

                bi_image = np.array(bi_image).astype(np.float32)
                ycbcr = convert_rgb_to_ycbcr(bi_image)

                y = ycbcr[..., 0]
                y /= 255.
                y = torch.from_numpy(y).to(device)
                y = y.unsqueeze(0).unsqueeze(0)


                with torch.no_grad():
                    preds = model(y).clamp(0.0, 1.0)
                orig=orig.view(1,1,orig.size()[0],orig.size()[1])
                #psnr = calc_psnr(orig, preds)
                #ssim = calc_ssim(orig, preds)



                #print('PSNR: {:.2f}'.format(psnr))
                #print('SSIM: {:.2f}'.format(ssim))

                preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

                output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
                output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
                output = pil_image.fromarray(output)
                output.save(op_name)
                #psnr_dict[file] = psnr.item()
                #ssim_dict[file] = ssim.item()
                #print(psnr_dict[file], ssim_dict[file])
                i+=1
        #end = time.time()
        print(i)
        '''timexec=end-start
        test_metrics={
            "psnr_vs_epoch":psnr_dict,
            "ssim_vs_epoch":ssim_dict,
            "time_of_execution":timexec
        }

        json_path=args.op_dir+"/test_metrics.json"
        with open(json_path, "w") as outfile:
            json.dump(test_metrics, outfile)'''
