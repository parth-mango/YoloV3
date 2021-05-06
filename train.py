import argparse
import cv2
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from utils.parse_config import *
import test  # import test.py to get mAP after each epoch
from model import *
from utils.datasets import *
from utils.utils import *
from yolo_decoder import *
from make_data import *
import pytorch_ssim
from planercnn.config import InferenceConfig
from planercnn.utils import *
from planercnn.visualize_utils import *
from planercnn.evaluate_utils import *
from planercnn.planercnn_decoder import *



mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    # print('Apex recommended for mixed precision and faster training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# Hyperparameters https://github.com/ultralytics/yolov3/issues/310

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])


def train(plane_parse_args,yolo_parse_args,midas_parse_args):
    #For yolo
    print("Opt", yolo_parse_args)
    opt= yolo_parse_args
    cfg = opt.cfg
    # print(opt)
    data = opt.data
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial yolo training weights
    opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
    # print("Opt img size", opt.img_size)
    imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)

    #For Planercnn
    opt_plane= plane_parse_args
    planercfg= InferenceConfig(opt_plane)
    

    #For Midas
    midas_args= midas_parse_args
    # Image Sizes
    gs = 64  # (pixels) grid size
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
    if opt.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = grid_min * gs, grid_max * gs
    img_size = imgsz_max  # initialize with max size

    # Configure run
    init_seeds()
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
    hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

    # Remove previous results
    for f in glob.glob('*_batch*.png') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Model_Head(cfg= cfg, planercfg=planercfg ).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    start_epoch = 0
    best_fitness = 0.0
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(weights, map_location=device)

        # load model
        try:
            # chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    elif len(weights) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        load_darknet_weights(model, weights)

    
    
    midas_params= torch.load(midas_args.weights) # Midas weights 
    
    if "optimizer" in midas_params:
        midas_params = midas_params["model"]

    model.load_state_dict(midas_params,strict=False)


    model.load_state_dict(torch.load(opt_plane.checkpoint_dir + '/checkpoint.pth'),strict=False)
    model.to(device)
    
    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    lf = lambda x: (((1 + math.cos(
        x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine https://arxiv.org/pdf/1812.01187.pdf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, [round(epochs * x) for x in [0.8, 0.9]], 0.1, start_epoch - 1)

    # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Datasets
    y_params_trn = dict(path = train_path, img_size = img_size, batch_size = batch_size, augment=True,hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)
    
    y_params_tst= dict(path = test_path, img_size = imgsz_test, batch_size = batch_size, augment=False,hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)
    
    m_params= None

    p_params= dict(options=opt_plane, config=planercfg,random=False)
    
    #Loading complete dataset
    training_dataset = load_data(p_params, y_params_trn, m_params)
    # print(len(training_dataset), "training dataset lenght")
    testing_dataset = load_data(p_params, y_params_tst,m_params)
                                  
    
    
    
#	midas_trn_dataset= midas_data(train_path)
    
#    midas_tst_dataset= midas_data(test_path)
    
    # Complete train loader
    batch_size = min(batch_size, len(training_dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    trainloader = torch.utils.data.DataLoader(training_dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=training_dataset.collate_fn)

    # Complete Testloader
    testloader = torch.utils.data.DataLoader(testing_dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=testing_dataset.collate_fn)
                                             
    

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(training_dataset.labels, nc).to(device)  # attach class weights

    # Model EMA
    ema = torch_utils.ModelEMA(model)

    # Start training
    nb = len(trainloader)  # number of batches
    # print(nb, "Hey sister")
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)

    loss_list=[]
   
    
    for epoch in range(start_epoch, epochs+1):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        # if dataset.image_weights:
        #     w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
        #     image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
        #     dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4).to(device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(trainloader))  # progress bar
        
        for i, (planer_data, yolo_data, midas_data) in pbar:  # batch -------------------------------------------------------------
            
            optimizer.zero_grad()
            imgs, targets, paths, _= yolo_data
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Burn-in
            if ni <= n_burn * 2:
                model.gr = np.interp(ni, [0, n_burn * 2], [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                if ni == n_burn:  # burnin complete
                    print_model_biases(model)

                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, [0, n_burn], [0.9, hyp['momentum']])

            # Multi-Scale training
            if opt.multi_scale:
                if ni / accumulate % 1 == 0:  # Â adjust img_size (67% - 150%) every 1 batch
                    img_size = random.randrange(grid_min, grid_max + 1) * gs
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                    
            yolo_input= imgs
            
            
            # from NVlabs Planercnn training 

            data_pair,plane_img,plane_np = planer_data
            # print(data_pair[0], "Planer Data")
            sampleIndex = i
            sample = data_pair

            plane_losses = []            

            input_pair = []
            detection_pair = []
            dicts_pair = []
            
            camera = sample[30][0].cuda()
            
            #for indexOffset in [0, ]:
            indexOffset=0
            images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda()
            # print(images, "Sample Image")  
            masks = (gt_segmentation == torch.arange(gt_segmentation.max() + 1).cuda().view(-1, 1, 1)).float()
            input_pair.append({'image': images, 'depth': gt_depth, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'camera': camera, 'plane': planes[0], 'masks': masks, 'mask': gt_masks})
            # print(input_pair[0]['image'], "Image input pair")
            plane_input= dict(input = [images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera], mode='inference_detection', use_nms=2, use_refinement=True, return_feature_map=False)






            # print(len(midas_data), "hello")


            depth_img_size,depth_img,depth_target = midas_data
            
            

            depth_sample = torch.from_numpy(depth_img).to(device).unsqueeze(0)
            
            # dp_array=np.transpose(depth_img,(2, 1,0))

            # print(dp_array.shape, "Depth image shape 369 train")
            
            
            
            # cv2.imwrite('depth_img4.jpg', dp_array)
            # midas_input= depth_sample

            # Run model
            plane_output, yolo_output,midas_output = model.forward(yolo_input, plane_input)
            print(midas_output.shape, "Midas output 369 train")
            pred = yolo_output
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, detections, detection_masks, detection_gt_parameters, detection_gt_masks, rpn_rois, roi_features, roi_indices, depth_np_pred = plane_output
            
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_parameter_loss = compute_losses(planercfg, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters)

            plane_losses =[rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + mrcnn_parameter_loss]


            if depth_np_pred.shape != gt_depth.shape:
                depth_np_pred = torch.nn.functional.interpolate(depth_np_pred.unsqueeze(1), size=(512, 512), mode='bilinear',align_corners=False).squeeze(1)
                pass

            if planercfg.PREDICT_NORMAL_NP:
                normal_np_pred = depth_np_pred[0, 1:]                    
                depth_np_pred = depth_np_pred[:, 0]
                gt_normal = gt_depth[0, 1:]                    
                gt_depth = gt_depth[:, 0]
                depth_np_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                normal_np_loss = l2LossMask(normal_np_pred[:, 80:560], gt_normal[:, 80:560], (torch.norm(gt_normal[:, 80:560], dim=0) > 1e-4).float())
                plane_losses.append(depth_np_loss)
                plane_losses.append(normal_np_loss)
            else:
                depth_np_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                plane_losses.append(depth_np_loss)
                normal_np_pred = None
                pass

            if len(detections) > 0:
                detections, detection_masks = unmoldDetections(planercfg, camera, detections, detection_masks, depth_np_pred, normal_np_pred, debug=False)
                if 'refine_only' in opt_plane.suffix:
                    detections, detection_masks = detections.detach(), detection_masks.detach()
                    pass
                XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(planercfg, camera, detections, detection_masks, depth_np_pred, return_individual=True)
                detection_mask = detection_mask.unsqueeze(0)                        
            else:
                XYZ_pred = torch.zeros((3, planercfg.IMAGE_MAX_DIM, planercfg.IMAGE_MAX_DIM)).cuda()
                detection_mask = torch.zeros((1, planercfg.IMAGE_MAX_DIM, planercfg.IMAGE_MAX_DIM)).cuda()
                plane_XYZ = torch.zeros((1, 3, planercfg.IMAGE_MAX_DIM, planercfg.IMAGE_MAX_DIM)).cuda() 
                detections = torch.zeros((3, planercfg.IMAGE_MAX_DIM, planercfg.IMAGE_MAX_DIM)).cuda()
                detection_masks = torch.zeros((3, planercfg.IMAGE_MAX_DIM, planercfg.IMAGE_MAX_DIM)).cuda()                        
                pass


            #input_pair.append({'image': images, 'depth': gt_depth, 'mask': gt_masks, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'parameters': detection_gt_parameters, 'plane': planes, 'camera': camera})
            detection_pair.append({'XYZ': XYZ_pred, 'depth': XYZ_pred[1:2], 'mask': detection_mask, 'detection': detections, 'masks': detection_masks, 'plane_XYZ': plane_XYZ, 'depth_np': depth_np_pred})

            loss_fn = nn.MSELoss()

            try:

                plane_parameters = torch.from_numpy(plane_np['plane_parameters']).cuda()
                plane_masks = torch.from_numpy(plane_np['plane_masks']).cuda()
                plane_parameters_pred = detection_pair[0]['detection'][:, 6:9]
                plane_masks_pred = detection_pair[0]['masks'][:, 80:560]

                if plane_parameters_pred.shape != plane_parameters.shape:
                    plane_parameters_pred = torch.nn.functional.interpolate(plane_parameters_pred.unsqueeze(1).unsqueeze(0), size=plane_parameters.shape, mode='bilinear',align_corners=True).squeeze()
                    pass
                if plane_masks_pred.shape != plane_masks.shape:
                    plane_masks_pred = torch.nn.functional.interpolate(plane_masks_pred.unsqueeze(1).unsqueeze(0), size=plane_masks.shape, mode='trilinear',align_corners=True).squeeze()
                    pass

                # print('plane_parameters',plane_parameters.shape)
                # print('plane_masks',plane_masks.shape)
                # print('plane_parameters_pred',plane_parameters_pred.shape)
                # print('plane_masks_pred',plane_masks_pred.shape)
               
                
                plane_params_loss = loss_fn(plane_parameters_pred,plane_parameters) + loss_fn(plane_masks_pred,plane_masks)
            except:
                plane_params_loss = 1

            #print('plane_params_loss',plane_params_loss)


            predicted_detection = visualizeBatchPair(opt_plane, planercfg, input_pair, detection_pair, indexOffset=i)
            # print(type(predicted_detection), "type")
            predicted_detection = torch.from_numpy(predicted_detection)

            # print(type(plane_img), "Plane img type")
            # plane_img = torch.from_numpy(plane_img)
            if predicted_detection.shape != plane_img.shape:
                predicted_detection = torch.nn.functional.interpolate(predicted_detection.permute(2,0,1).unsqueeze(0).unsqueeze(1), size=plane_img.permute(2,0,1).shape).squeeze()
                pass

            plane_img = plane_img.permute(2,0,1)

            #print('plane_img',plane_img.shape)
            #print('predicted_detection',predicted_detection.shape)

            plane_loss_ssim = pytorch_ssim.SSIM() #https://github.com/Po-Hsun-Su/pytorch-ssim
            
            
            pln_ssim = torch.clamp(1-plane_loss_ssim(predicted_detection.unsqueeze(0).type(torch.cuda.FloatTensor),plane_img.unsqueeze(0).type(torch.cuda.FloatTensor)),min=0,max=1) #https://github.com/jorge-pessoa/pytorch-msssim
            
            plane_loss = sum(plane_losses) + pln_ssim
            # print(plane_loss, "plane loss")
            plane_losses = [l.data.item() for l in plane_losses] 


            depth_prediction = midas_output
            
            depth_prediction = (
                            torch.nn.functional.interpolate(
                                depth_prediction.unsqueeze(1),
                                size=tuple(depth_img_size[:2]),
                                mode="bicubic",
                                align_corners=False,
                            )
                            #.unsqueeze(0)
                            #.cpu()
                            #.numpy()
                            )

            # print(depth_prediction.shape, "Depth prediction type 484 train")
            
            # dp_array=depth_prediction.cpu().detach().numpy()

            # print(dp_array.shape, "Depth prediction type 488 train")
            
            # cv2.imwrite('depth_img1.jpg', dp_array)

            bits=2

            depth_min = depth_prediction.min()
            depth_max = depth_prediction.max()

            max_val = (2**(8*bits))-1

            if depth_max - depth_min > np.finfo("float").eps:
                depth_out = max_val * (depth_prediction - depth_min) / (depth_max - depth_min)
                # print(type(depth_out), "type")
            else:
                depth_out= torch.tensor(0.0)
            # print('depth_target',depth_target.shape)
            depth_target = torch.from_numpy(np.asarray(depth_target)).to(device).type(torch.cuda.FloatTensor).unsqueeze(0)

            depth_target = (
                torch.nn.functional.interpolate(
                    depth_target.unsqueeze(1),
                    size=depth_img_size[:2],
                    mode="bicubic",
                    align_corners=False
                )
                #.unsqueeze(0)
                #.cpu()
                #.numpy()
                )
            # print(depth_target.shape, depth_prediction.shape, "hey man")
            #Computing depth loss
            # print(depth_out, "Depth_out")
            depth_pred = Variable( depth_out,  requires_grad=True)
            depth_target = Variable( depth_target, requires_grad = False)
            
            # loss_fn = nn.MSELoss()
            # print(len(depth_pred), len(depth_target), "length depth")
            loss_ssim = pytorch_ssim.SSIM()
            
            ssim_out = torch.clamp(1-loss_ssim(depth_pred,depth_target),min=0,max=1)
            
            
            
            loss_RMSE = torch.sqrt(loss_fn(depth_pred, depth_target))
            
            depth_loss = (0.0001*loss_RMSE) + ssim_out

            # print(depth_loss, "depth loss")
            



            # Compute loss
            # print(len(pred), len(targets), "length")
            y_loss, y_loss_items = compute_loss(pred, targets, model)
            # print(y_loss, "yolo loss")
            if not torch.isfinite(y_loss):
                print('WARNING: non-finite loss, ending training ', y_loss_items)
                return results

            # Scale loss by nominal batch_size of 64
#            y_loss *= batch_size / 64
            
            #Total Loss
            total_loss= y_loss + plane_loss + depth_loss 
            print(total_loss, "total_loss")

            # Compute gradient
            print(mixed_precision, "mixed precision")
            if mixed_precision:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    print("i m in mixed precision")
            else:
                total_loss.backward()

            # Optimize accumulated gradient
            #if ni % accumulate == 0:
            optimizer.step()
             #   optimizer.zero_grad()
            ema.update(model)

            # Print batch results
            mloss = (mloss * i + y_loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            # Plot images with bounding boxes
#           if ni < 1:
#                f = 'train_batch%g.png' % i  # filename
#                plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)
#               if tb_writer:
#                    tb_writer.add_image(f, cv2.imread(f)[:, :, ::-1], dataformats='HWC')
#                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        # if not opt.notest or final_epoch:  # Calculate mAP
        #     is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
        #     results, maps = test.test(cfg,
        #                                 data,
        #                                 batch_size=batch_size,
        #                                 img_size=imgsz_test,
        #                                 model=ema.ema,
        #                                 save_json=final_epoch and is_coco,
        #                                 single_cls=opt.single_cls,
        #                                 dataloader=testloader)

#        # Write epoch results
#        with open(results_file, 'a') as f:
#            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
#        if len(opt.name) and opt.bucket:
#           os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # Write Tensorboard results
#        if tb_writer:
#            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
#                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
#                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
#            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
#               tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
#        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
#        if fi > best_fitness:
#            best_fitness = fi

        # Save training results
#        save = (not opt.nosave) or (final_epoch and not opt.evolve)
#        if save:
#            with open(results_file, 'r') as f:
        # Create checkpoint
        chkpt = {'epoch': epoch,
                'best_loss': best_loss,
                'training_results': f.read(),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}

        # Save last checkpoint
        torch.save(chkpt, last)

        # Save best checkpoint
        is_best= False
        if (all_loss < best_loss) and not final_epoch:
            best_loss =all_loss
            is_best= True
            torch.save(chkpt, best)

        # Save backup every 10 epochs (optional)
        # if epoch > 0 and epoch % 10 == 0:
        #     torch.save(chkpt, wdir + 'backup%g.pt' % epoch)

        # Delete checkpoint
        # del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------

    # end training
    # n = opt.name
    # if len(n):
    #     n = '_' + n if not n.isnumeric() else n
    #     fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
    #     for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
    #         if os.path.exists(f1):
    #             os.rename(f1, f2)  # rename
    #             ispt = f2.endswith('.pt')  # is *.pt
    #             strip_optimizer(f2) if ispt else None  # strip optimizer
    #             os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

        loss_list.append(total_loss.item())

    plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return loss_list


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epochs', type=int, default=300)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
#     parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
#     parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
#     parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
#     parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
#     parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
#     parser.add_argument('--img-size', nargs='+', type=int, default=[512], help='[min_train, max-train, test] img sizes')
#     parser.add_argument('--rect', action='store_true', help='rectangular training')
#     parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
#     parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
#     parser.add_argument('--notest', action='store_true', help='only test final epoch')
#     parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
#     parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
#     parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
#     parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='initial weights path')
#     parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
#     parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
#     parser.add_argument('--adam', action='store_true', help='use adam optimizer')
#     parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
#     parser.add_argument('--input', type=str, default='input', help=' input path')
#     parser.add_argument('--output', type=str, default='output', help='output path')
#     parser.add_argument('--midasweights', type=str, default='/content/YoloV3/midas/model-f6b98070.pt', help='initial weights path')
#     opt = parser.parse_args()
#     opt.weights = last if opt.resume else opt.weights
#     print(opt)

# opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
# device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
# if device.type == 'cpu':
#     mixed_precision = False

#     train()  # train normally
# else:
#     train()  # train normally

    # scale hyp['obj'] by img_size (evolved at 320)
    # hyp['obj'] *= opt.img_size[0] / 320.

    # tb_writer = None
    # if not opt.evolve:  # Train normally
    #     try:
    #         # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
    #         from torch.utils.tensorboard import SummaryWriter

    #         tb_writer = SummaryWriter()
    #         print("Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/")
    #     except:
    #         pass

        

    # else:  # Evolve hyperparameters (optional)
    #     opt.notest, opt.nosave = True, True  # only test/save final epoch
    #     if opt.bucket:
    #         os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

    #     for _ in range(1):  # generations to evolve
    #         if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
    #             # Select parent(s)
    #             parent = 'single'  # parent selection method: 'single' or 'weighted'
    #             x = np.loadtxt('evolve.txt', ndmin=2)
    #             n = min(5, len(x))  # number of previous results to consider
    #             x = x[np.argsort(-fitness(x))][:n]  # top n mutations
    #             w = fitness(x) - fitness(x).min()  # weights
    #             if parent == 'single' or len(x) == 1:
    #                 # x = x[random.randint(0, n - 1)]  # random selection
    #                 x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
    #             elif parent == 'weighted':
    #                 x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

    #             # Mutate
    #             method, mp, s = 3, 0.9, 0.2  # method, mutation probability, sigma
    #             npr = np.random
    #             npr.seed(int(time.time()))
    #             g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
    #             ng = len(g)
    #             if method == 1:
    #                 v = (npr.randn(ng) * npr.random() * g * s + 1) ** 2.0
    #             elif method == 2:
    #                 v = (npr.randn(ng) * npr.random(ng) * g * s + 1) ** 2.0
    #             elif method == 3:
    #                 v = np.ones(ng)
    #                 while all(v == 1):  # mutate until a change occurs (prevent duplicates)
    #                     # v = (g * (npr.random(ng) < mp) * npr.randn(ng) * s + 1) ** 2.0
    #                     v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
    #             for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
    #                 hyp[k] = x[i + 7] * v[i]  # mutate

    #         # Clip to limits
    #         keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
    #         limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
    #         for k, v in zip(keys, limits):
    #             hyp[k] = np.clip(hyp[k], v[0], v[1])

    #         # Train mutation
    #         results = train()

    #         # Write mutation results
    #         print_mutation(hyp, results, opt.bucket)

    #         # Plot results
    #         # plot_evolution_results(hyp)
