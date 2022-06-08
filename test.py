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
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--test-img-dir', type=str, required=True)
    parser.add_argument('--op-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()
    ip_path=args.test_img_dir

    filenames=os.listdir(ip_path)
    
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
    psnr_dict={}
    ssim_dict={}
    bi_psnr_dict={}
    bi_ssim_dict={}
    
    # we will save the conv layer weights in this list
    model_weights =[]
    #we will save the 49 conv layers in this list
    conv_layers = []
    # get all the model children as list
    model_children = list(model.children())
    #counter to keep count of the conv layers
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolution layers: {counter}")
    print("conv_layers")
    for file in filenames:
          image = pil_image.open(os.path.join(ip_path,file)).convert('RGB')

          orig= np.array(image).astype(np.float32)
          orig=convert_rgb_to_ycbcr(orig)  
          orig = orig[..., 0]
          orig /= 255.
          orig = torch.from_numpy(orig).to(device)
          oig = orig.unsqueeze(0).unsqueeze(0)


          image_width = (image.width // args.scale) * args.scale
          image_height = (image.height // args.scale) * args.scale
          image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
          image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
          op_name=os.path.join(args.op_dir,file)  
          image.save(op_name.replace('.', '_downsized_x{}.'.format(args.scale)))
          image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
          
          image.save(op_name.replace('.', '_bicubic_x{}.'.format(args.scale)))

          image = np.array(image).astype(np.float32)
          ycbcr = convert_rgb_to_ycbcr(image)

          y = ycbcr[..., 0]
          y /= 255.
          y = torch.from_numpy(y).to(device)
          y = y.unsqueeze(0).unsqueeze(0)
          outputs = []
          names = []
          for layer in conv_layers[0:]:
              image = layer(y)
              outputs.append(y)
              names.append(str(y))
          print(len(outputs))
          #print feature_maps
          for feature_map in outputs:
              print(feature_map.shape)
                
          processed = []
          for feature_map in outputs:
              feature_map = feature_map.squeeze(0)
              gray_scale = torch.sum(feature_map,0)
              gray_scale = gray_scale / feature_map.shape[0]
              processed.append(gray_scale.data.cpu().numpy())
          for fm in processed:
              print(fm.shape)
          
          fig = plt.figure(figsize=(30, 50))
          for i in range(len(processed)):
              a = fig.add_subplot(5, 4, i+1)
              imgplot = plt.imshow(processed[i])
              a.axis("off")
              a.set_title(names[i].split('(')[0], fontsize=30)
          plt.savefig(os.path.join(args.op_dir,'feature_maps.jpg'), bbox_inches='tight')
        
          with torch.no_grad():
              preds = model(y).clamp(0.0, 1.0)
          orig=orig.view(1,1,orig.size()[0],orig.size()[1])
          psnr = calc_psnr(orig, preds)
          ssim = calc_ssim(orig, preds)
          bi_psnr = calc_psnr(orig, y)
          bi_ssim = calc_ssim(orig, y)
          
          
          print('PSNR: {:.2f}'.format(psnr))
          print('SSIM: {:.2f}'.format(ssim))

          preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

          output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
          output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
          output = pil_image.fromarray(output)
          output.save(op_name.replace('.', '_srcnn_x{}.'.format(args.scale)))
          psnr_dict[file] = psnr.item()
          ssim_dict[file] = ssim.item()
          bi_psnr_dict[file] = bi_psnr.item()
          bi_ssim_dict[file] = bi_ssim.item()
          #print(psnr_dict[file], ssim_dict[file])
    end = time.time()
    timexec=end-start
    test_metrics={
        "psnr_vs_epoch":psnr_dict,
        "ssim_vs_epoch":ssim_dict,
        "bicubic_psnr_vs_epoch":bi_psnr_dict,
        "bicubic_ssim_vs_epoch":bi_ssim_dict,
        "time_of_execution":timexec
    }

    json_path=args.op_dir+"/test_metrics.json"
    with open(json_path, "w") as outfile:
        json.dump(test_metrics, outfile)
