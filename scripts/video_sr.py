"""
Modified from [CodeFormer](https://github.com/sczhou/CodeFormer).
When using or redistributing this feature, please comply with the [S-Lab License 1.0](https://github.com/sczhou/CodeFormer?tab=License-1-ov-file).
We kindly request that you respect the terms of this license in any usage or redistribution of this component.
"""

import os
import cv2
import argparse
import glob
import sys

import torch
from torchvision.transforms.functional import normalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray

from basicsr.utils.registry import ARCH_REGISTRY


def set_realesrgan():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    if torch.cuda.is_available(): # set False in CPU/MPS mode
        no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
        if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="./pretrained_models/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=args.bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )

    if not gpu_is_available():  # CPU
        import warnings
        warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                        'The unoptimized RealESRGAN is slow on CPU. '
                        'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                        category=RuntimeWarning)
    return upsampler




if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = get_device()
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, help='Input video')
    parser.add_argument('-o', '--output_path', type=str, default=None, 
            help='Output folder')
    parser.add_argument('-w', '--fidelity_weight', type=float, default=0.5, 
            help='Balance the quality and fidelity. Default: 0.5')
    parser.add_argument('-s', '--upscale', type=int, default=2, 
            help='The final upsampling scale of the image. Default: 2')
    parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces. Default: False')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face. Default: False')
    parser.add_argument('--draw_box', action='store_true', help='Draw the bounding box for the detected faces. Default: False')
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    parser.add_argument('--detection_model', type=str, default='retinaface_resnet50', 
            help='Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n. \
                Default: retinaface_resnet50')
    parser.add_argument('--bg_upsampler', type=str, default='None', help='Background upsampler. Optional: realesrgan')
    parser.add_argument('--face_upsample', action='store_true', help='Face upsampler after enhancement. Default: False')
    parser.add_argument('--bg_tile', type=int, default=400, help='Tile size for background sampler. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces. Default: None')
    
    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    w = args.fidelity_weight
    input_video = False
    if args.input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        from basicsr.utils.video_util import VideoReader, VideoWriter
        input_img_list = []
        vidreader = VideoReader(args.input_path)
        image = vidreader.get_frame()
        while image is not None:
            input_img_list.append(image)
            image = vidreader.get_frame()
        audio = vidreader.get_audio()
        fps = vidreader.get_fps()    
        video_name = os.path.basename(args.input_path)[:-4]
        result_root = f'./hq_results/{video_name}_{w}_{args.upscale}'
        input_video = True
        vidreader.close()
    else: 
        raise RuntimeError("input should be mp4 file")

    if not args.output_path is None: # set output path
        result_root = args.output_path

    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError('No input image/video is found...\n' 
            '\tNote that --input_path for video should end with .mp4|.mov|.avi')

    # ------------------ set up background upsampler ------------------
    if args.bg_upsampler == 'realesrgan':
        bg_upsampler = set_realesrgan()
    else:
        bg_upsampler = None

    # ------------------ set up face upsampler ------------------
    if args.face_upsample:
        if bg_upsampler is not None:
            face_upsampler = bg_upsampler
        else:
            face_upsampler = set_realesrgan()
    else:
        face_upsampler = None

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128', '256']).to(device)
    
    ckpt_path = './pretrained_models/hallo2/net_g.pth'
    
    checkpoint = torch.load(ckpt_path)['params_ema']
    m, n = net.load_state_dict(checkpoint, strict=False)
    print("missing key: ", m)
    assert len(n)==0
    net.eval()

    # ------------------ set up FaceRestoreHelper -------------------
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    if not args.has_aligned: 
        print(f'Face detection model: {args.detection_model}')
    if bg_upsampler is not None: 
        print(f'Background upsampling: True, Face upsampling: {args.face_upsample}')
    else:
        print(f'Background upsampling: False, Face upsampling: {args.face_upsample}')

    face_helper = FaceRestoreHelper(
        args.upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model = args.detection_model,
        save_ext='png',
        use_parse=True,
        device=device)
    
    n = -1
    input_img_list = input_img_list[:n]
    length = len(input_img_list)

    overlay = 4
    chunk = 16
    idx_list = []

    i=0
    j=0
    while i < length and j < length:
        j = min(i+chunk, length)
        idx_list.append([i, j])
        i = j-overlay
        

    id_list = []

    # -------------------- start to processing ---------------------
    for i, idx in enumerate(idx_list):
        # clean all the intermediate results to process the next image
        face_helper.clean_all()

        start = idx[0]
        end = idx[1]

        img_list = input_img_list[start:end]

        for j, img_path in enumerate(img_list):
        
            if isinstance(img_path, str):
                img_name = os.path.basename(img_path)
                basename, ext = os.path.splitext(img_name)
                print(f'[{j+1}/{chunk}] Processing: {img_name}')
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            else: # for video processing
                basename = str(i).zfill(4)
                img_name = f'{video_name}_{basename}_{j}' if input_video else basename
                print(f'[{j+1}/{chunk}] Processing: {img_name}')
                img = img_path

            if args.has_aligned: 
                # the input faces are already cropped and aligned
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_helper.is_gray = is_gray(img, threshold=10)
                if face_helper.is_gray:
                    print('Grayscale input: True')
                face_helper.cropped_faces = [img]
            else:
                face_helper.read_image(img)
                # get face landmarks for each face
                num_det_faces = face_helper.get_face_landmarks_5(
                    only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5)
                print(f'\tdetect {num_det_faces} faces')
                # align and warp each face
                face_helper.align_warp_face()    

        crop_image = []
        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0)

            crop_image.append(cropped_face_t)
        
        assert len(crop_image)==len(img_list)

        crop_image = torch.cat(crop_image, dim=0).to(device)
        crop_image = crop_image.unsqueeze(0)

        output, top_idx = net.inference(crop_image, w=w, adain=True)
        assert output.shape==crop_image.shape

        for k in range(output.shape[1]):
            face_output = output[:, k:k+1]
            restored_face = tensor2img(face_output.squeeze_(1), rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            cropped_face = face_helper.cropped_faces[k]
            face_helper.add_restored_face(restored_face, cropped_face)

        bg_img_list = []
        # paste_back
        if not args.has_aligned:
            for img in img_list:
                # upsample the background
                if bg_upsampler is not None:
                    # Now only support RealESRGAN for upsampling background
                    bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
                else:
                    bg_img = None
                bg_img_list.append(bg_img)


            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if args.face_upsample and face_upsampler is not None: 
                restored_img_list = face_helper.paste_faces_to_input_image(upsample_img_list=bg_img_list, draw_box=args.draw_box, face_upsampler=face_upsampler)
            else:
                restored_img_list = face_helper.paste_faces_to_input_image(upsample_img_list=bg_img_list, draw_box=args.draw_box)

        torch.cuda.empty_cache()
        
        if i!=0:
            restored_img_list = restored_img_list[overlay:]
        

        # save restored img
        if not args.has_aligned and len(restored_img_list)!=0:
            if args.suffix is not None:
                basename = f'{video_name}_{args.suffix}_{i}'
            for k, restored_img in enumerate(restored_img_list): 
                kk = str(k).zfill(3)
                save_restore_path = os.path.join(result_root, 'final_results', f'{basename}_{kk}.png')
                imwrite(restored_img, save_restore_path)

    # save enhanced video
    if input_video:
        print('Video Saving...')
        # load images
        video_frames = []
        img_list = sorted(glob.glob(os.path.join(result_root, 'final_results', '*.[jp][pn]g')))

        assert len(img_list)==length, print(len(img_list), length)

        # write images to video
        sample_img = cv2.imread(img_list[0])
        height, width = sample_img.shape[:2]

        if args.suffix is not None:
            video_name = f'{video_name}_{args.suffix}.png'
        save_restore_path = os.path.join(result_root, f'{video_name}.mp4')

        vidwriter = VideoWriter(save_restore_path, height, width, fps, audio)
         
        for img_path in img_list:
            print(img_path)
            img = cv2.imread(img_path)
            vidwriter.write_frame(img)

        vidwriter.close()

    print(f'\nAll results are saved in {result_root}')
