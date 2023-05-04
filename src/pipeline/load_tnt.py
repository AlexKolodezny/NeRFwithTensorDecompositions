import os
import numpy as np
import torch
from PIL import Image
from itertools import islice

def open_file(pth, mode='r'):
    return open(pth, mode=mode)

def listdir(pth):
    return os.listdir(pth)

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()

rot_z = lambda z: torch.Tensor([
    [np.cos(z), -np.sin(z), 0, 0],
    [np.sin(z), np.cos(z), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def intrinsic_matrix(fx: float,
                     fy: float,
                     cx: float,
                     cy: float,
                     xnp = np):
    """Intrinsic matrix for a pinhole camera in OpenCV coordinate system."""
    return xnp.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.],
    ])


def get_pixtocam(focal: float,
                 width: float,
                 height: float,
                 xnp = np):
    """Inverse intrinsic matrix for a perfect pinhole camera."""
    camtopix = intrinsic_matrix(focal, focal, width * .5, height * .5, xnp)
    return xnp.linalg.inv(camtopix)


def load_tnt_data(
        data_dir, image_downscale_factor=1, image_downscale_filter='area',
        testskip=1, scene_scale=2 / 3, scene_rot_z_deg=0
):
    splits = ['train', 'test']

    all_imgs = []
    all_poses = []
    counts = [0]

    for split_str in splits:
      basedir = os.path.join(data_dir, split_str)

      if split_str == 'train' or testskip == 0:
          skip = 1
      else:
          skip = testskip

      def load_files(dirname, load_fn, shape=None):
          files = [
              os.path.join(basedir, dirname, f)
              for f in islice(sorted(listdir(os.path.join(basedir, dirname))), 0, None, skip)
          ]
          mats = np.array([load_fn(open_file(f, 'rb')) for f in files])
          if shape is not None:
              mats = mats.reshape(mats.shape[:1] + shape)
          return mats

      poses = load_files('pose', np.loadtxt, (4, 4))
      # Flip Y and Z axes to get correct coordinate frame.
      poses = np.matmul(poses, np.diag(np.array([1, -1, -1, 1])))

      # For now, ignore all but the first focal length in intrinsics
      intrinsics = load_files('intrinsics', np.loadtxt, (4, 4))

      images = load_files('rgb', lambda f: np.array(Image.open(f))) / 255.
      all_imgs.append(images)
      all_poses.append(poses)
      counts.append(counts[-1] + images.shape[0])

      focal = intrinsics[0, 0, 0]

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(counts) - 1)]
    images = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = images.shape[1:3]
    pixtocams = get_pixtocam(focal, W, H)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
    poses[:, :3, 3] *= scene_scale
    render_poses[:, :3, 3] *= scene_scale

    if image_downscale_factor != 1:
        if type(image_downscale_factor) is not int or image_downscale_factor < 2 \
                or image_downscale_factor & (image_downscale_factor - 1) != 0:
            raise ValueError(f'Invalid {image_downscale_factor=}')
        if H % image_downscale_factor != 0 or W % image_downscale_factor != 0:
            raise ValueError(f'Invalid {image_downscale_factor=} with {W=}, {H=}')

        H = H // image_downscale_factor
        W = W // image_downscale_factor
        focal = focal / image_downscale_factor

        imgs_small = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            if image_downscale_filter == 'area':
                img = np.reshape(img, (H, image_downscale_factor, W, image_downscale_factor, 4))
                img = np.mean(img, axis=(1, 3))
                imgs_small[i] = img
            elif image_downscale_filter == 'antialias':
                imgs_small[i] = skimage.transform.resize(img, (H, W), anti_aliasing=True)
            else:
                raise ValueError(f'Unknown interpolation filter: {image_downscale_filter}')
        imgs = imgs_small

    return images, poses, render_poses, [H, W, focal], i_split