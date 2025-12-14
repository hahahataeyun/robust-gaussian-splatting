#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import time
import math

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

try:
    from simple_knn._C import knn_idx as fast_knn_idx
    FAST_KNN_AVAILABLE = True
    print("Fast knn module found for convergence loss")
except Exception:
    FAST_KNN_AVAILABLE = False
    print("Fast knn module not found for convergence loss, using torch cdist")

def init_wandb(dataset, opt):
    
    if not WANDB_AVAILABLE:
        print("wandb not available: skipping wandb logging.")
        return None
    config = {
        "source_path": getattr(dataset, "source_path", None),
        "model_path": getattr(dataset, "model_path", None),
        "lambda_converge": getattr(opt, "lambda_converge", None),
        "converge_knn": getattr(opt, "converge_knn", None),
        "converge_interval": getattr(opt, "converge_interval", None),
        "merge_interval": getattr(opt, "merge_interval", None),
    }
    try:
        
        if opt.lambda_converge == 0:
            run_name = 'lc_' + os.path.basename(getattr(dataset, "source_path")) + "_noconv"
        else:
            if opt.merge_interval == 0:
                run_name = ('lc_' + os.path.basename(getattr(dataset, "source_path")) + 
                            "_lambda_" + str(opt.lambda_converge) + 
                            "_knn_" + str(opt.converge_knn) + 
                            "_convinterval_" + str(opt.converge_interval)
                            + "_nomerge")
            else:
                run_name = ('lc_' + os.path.basename(getattr(dataset, "source_path")) + 
                            "_lambda_" + str(opt.lambda_converge) + 
                            "_knn_" + str(opt.converge_knn) + 
                            "_convinterval_" + str(opt.converge_interval)
                            + "_mergeinterval_" + str(opt.merge_interval))

        return wandb.init(
            project="3dgs_convergence_regularization",
            name= run_name,
            config=config,
            dir=getattr(dataset, "model_path", None),
        )
    except Exception as exc:
        print(f"Failed to initialize wandb: {exc}")
        return None

def get_current_lambda_conv(step, target_lambda, decay_start, total_steps):
    if step < 3000:
        return 0.0
    elif step < decay_start:
        return target_lambda
    elif step <= total_steps:
        progress = (step - decay_start) / (total_steps - decay_start)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return target_lambda * cosine_decay

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    start_time = time.time()
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    wandb_run = init_wandb(dataset, opt)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss_before_converge = loss.detach()
        converge_loss_value = None
        
        # Convergence regularization: encourage visible Gaussians to move closer to their local neighbors
        # This is computed only on the set of Gaussians that were visible in the current render (visibility_filter)
        # to limit cost. The term is: mean(||x_i - mean(neighbors(x_i))||^2)
        target_lambda_converge = getattr(opt, 'lambda_converge', 0.0)

        if iteration > 3000 and target_lambda_converge > 0:
            converge_interval = int(getattr(opt, 'converge_interval', 10))
            if converge_interval > 0 and iteration % converge_interval == 0:
                try:
                    vis_filter = visibility_filter
                    if vis_filter is not None:
                        xyz_all = gaussians.get_xyz # (N, 3) (float32 on CUDA)
                        # Renderer returns indices (nonzero) possibly on CPU; move/flatten for consistent indexing
                        vis_filter = vis_filter.to(xyz_all.device, non_blocking=True)
                        visible_xyz = None
                        if vis_filter.dtype == torch.bool:
                            visible_xyz = xyz_all[vis_filter] # (V, 3) where V = number of visible points
                        else:
                            vis_indices = vis_filter.reshape(-1)
                            if vis_indices.numel() > 0:
                                vis_indices = vis_indices.long()
                                visible_xyz = xyz_all.index_select(0, vis_indices)
                        if visible_xyz is not None and visible_xyz.shape[0] > 1:
                            visible_xyz = visible_xyz.reshape(visible_xyz.shape[0], -1).contiguous() # (V, 3)
                            max_points = int(getattr(opt, 'converge_max_points', 4096))
                            if max_points > 0 and visible_xyz.shape[0] > max_points:
                                perm = torch.randperm(visible_xyz.shape[0], device=visible_xyz.device) # if too many points, randomly subsample
                                visible_xyz = visible_xyz.index_select(0, perm[:max_points]) # (max_points, 3)
                            V = visible_xyz.shape[0] # number of visible points considered
                            if V > 1:
                                k = min(int(getattr(opt, 'converge_knn', 5)), V - 1) # number of neighbors
                                if k > 0:
                                    neighbors = None
                                    if FAST_KNN_AVAILABLE:
                                        print("Using fast knn for convergence loss")
                                        try:
                                            knn_idx = fast_knn_idx(visible_xyz, k)
                                            neighbors = visible_xyz[knn_idx]
                                        except Exception:
                                            neighbors = None
                                    if neighbors is None:
                                        dists = torch.cdist(visible_xyz, visible_xyz, p=2)
                                        dists.fill_diagonal_(float('inf'))
                                        knn_idx = torch.topk(dists, k=k, largest=False).indices
                                        neighbors = visible_xyz[knn_idx]
                                    neighbor_mean = neighbors.mean(dim=1)
                                    converge_loss = (visible_xyz - neighbor_mean).pow(2).sum(dim=1).mean()
                                    converge_loss_value = converge_loss.detach()

                                    current_lambda_conv = get_current_lambda_conv(iteration, target_lambda_converge, 20000, getattr(opt, "iterations", 30_000))
                                    loss = loss + current_lambda_conv * converge_loss
                except Exception:
                    # fail-safe: if anything goes wrong (shapes, cuda) skip convergence loss for this iter
                    pass
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if wandb_run:
                wandb_log = {
                    "loss/before_converge": loss_before_converge.item(),
                    "loss/total": loss.item(),
                    "scene/num_gaussians": gaussians.get_xyz.shape[0],
                }
                if converge_loss_value is not None:
                    wandb_log["loss/converge"] = converge_loss_value.item()
                wandb_run.log(wandb_log, step=iteration)

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # Perform densification and pruning every densification_interval iterations after densify_from_iter
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Merge nearby, similar Gaussians every merge_interval iterations
            if getattr(opt, "merge_interval", 0) > 0 and iteration % opt.merge_interval == 0 and iteration < 7_000:
                if opt.merge_distance_threshold > 0 and opt.merge_sh_threshold > 0:
                    merges_done = gaussians.merge_close_gaussians(
                        opt.merge_distance_threshold,
                        opt.merge_sh_threshold,
                        max_merges=getattr(opt, "merge_max_per_iter", 1),
                        chunk_size=getattr(opt, "merge_chunk_size", 4096),
                    )
                    if wandb_run and merges_done > 0:
                        wandb_run.log({"scene/merges": merges_done}, step=iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    end_time = time.time()

    wandb_run.summary["total_training_time_seconds"] = end_time - start_time
    wandb_run.summary["total_number_of_gaussians"] = scene.gaussians.get_xyz.shape[0]
    if wandb_run:
        wandb_run.finish()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        # object = args.source_path.split(os.sep)[-1]

        # args.model_path = os.path.join("./output/", object, "lambda_convergence_", str(args.lambda_converge), "knn_", str(args.converge_knn), "_interval_", str(args.converge_interval))
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
    
