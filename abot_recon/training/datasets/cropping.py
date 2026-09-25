# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# croppping utilities
# --------------------------------------------------------
import PIL.Image
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa
import numpy as np  # noqa
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC



def colmap_to_opencv_intrinsics(matrix):
    matrix = matrix.copy()
    matrix[0, 2] -= 0.5
    matrix[1, 2] -= 0.5
    return matrix


def opencv_to_colmap_intrinsics(matrix):
    matrix = matrix.copy()
    matrix[0, 2] += 0.5
    matrix[1, 2] += 0.5
    return matrix

class ImageList:
    """ Convenience class to aply the same operation to a whole set of images.
    """

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch('resize', *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch('crop', *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


def rescale_image_depthmap(image, depthmap, camera_intrinsics, output_resolution, force=True, normal=None, far_mask=None):
    """ Jointly rescale a (image, depthmap) 
        so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        # can also use this with masks instead of depthmaps
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    # define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:  # image is already smaller than what is asked
        return (image.to_pil(), depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # first rescale the image so that it contains the crop
    image = image.resize(tuple(output_resolution), resample=lanczos if scale_final < 1 else bicubic)
    if depthmap is not None:
        depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)
        
    if normal is not None:
        normal = cv2.resize(normal, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)
    if far_mask is not None:
        far_mask = cv2.resize(far_mask, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

    return image.to_pil(), depthmap, camera_intrinsics, normal, far_mask


def resize_to_width_and_crop_or_pad(
    image,
    depthmap,
    camera_intrinsics,
    output_resolution,
    *,
    pad_rgb=(0.485, 0.456, 0.406),
    min_vertical_pad_px=0,
    normal=None,
    far_mask=None,
):
    """Fit width exactly, then center-crop or center-pad only in height.

    RGB, depth, optional normal and far-mask, and camera intrinsics undergo the
    same raster transform. The returned ``content_bbox`` marks the real image
    area in the final canvas so callers can keep photometric augmentation away
    from synthetic mean padding.
    """
    image = ImageList(image)
    input_width, input_height = (int(v) for v in image.size)
    out_width, out_height = (int(v) for v in output_resolution)
    if input_width <= 0 or input_height <= 0:
        raise ValueError(f"input image size must be positive, got {image.size}")
    if out_width <= 0 or out_height <= 0:
        raise ValueError(
            f"output resolution must be positive, got {(out_width, out_height)}"
        )
    try:
        min_vertical_pad_px_numeric = float(min_vertical_pad_px)
        min_vertical_pad_px = int(min_vertical_pad_px_numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("min_vertical_pad_px must be a non-negative integer") from exc
    if (
        not np.isfinite(min_vertical_pad_px_numeric)
        or min_vertical_pad_px_numeric != min_vertical_pad_px
        or min_vertical_pad_px < 0
    ):
        raise ValueError(
            "min_vertical_pad_px must be a non-negative integer, got "
            f"{min_vertical_pad_px_numeric!r}"
        )
    if 2 * min_vertical_pad_px >= out_height:
        raise ValueError(
            "min_vertical_pad_px leaves no real image content: "
            f"pad={min_vertical_pad_px}, output_height={out_height}"
        )
    if depthmap is not None:
        assert tuple(depthmap.shape[:2]) == (input_height, input_width)
    if normal is not None:
        assert tuple(normal.shape[:2]) == (input_height, input_width)
    if far_mask is not None:
        assert tuple(far_mask.shape[:2]) == (input_height, input_width)

    pad_rgb = np.asarray(pad_rgb, dtype=np.float64).reshape(-1)
    if pad_rgb.size == 1:
        pad_rgb = np.repeat(pad_rgb, 3)
    if pad_rgb.size != 3 or not np.isfinite(pad_rgb).all():
        raise ValueError(f"pad_rgb must contain one or three finite values, got {pad_rgb}")
    if np.any(pad_rgb < 0.0) or np.any(pad_rgb > 1.0):
        raise ValueError(f"pad_rgb must be in [0, 1], got {pad_rgb}")

    width_scale = out_width / input_width
    resized_height = max(1, int(np.round(input_height * width_scale)))
    resized_resolution = (out_width, resized_height)
    image = image.resize(
        resized_resolution,
        resample=lanczos if width_scale < 1.0 else bicubic,
    )
    if depthmap is not None:
        depthmap = cv2.resize(
            depthmap,
            resized_resolution,
            interpolation=cv2.INTER_NEAREST,
        )
    if normal is not None:
        normal = cv2.resize(
            normal,
            resized_resolution,
            interpolation=cv2.INTER_NEAREST,
        )
    if far_mask is not None:
        far_mask_dtype = far_mask.dtype
        far_mask_resize_input = (
            far_mask.astype(np.uint8, copy=False)
            if far_mask_dtype == np.bool_
            else far_mask
        )
        far_mask = cv2.resize(
            far_mask_resize_input,
            resized_resolution,
            interpolation=cv2.INTER_NEAREST,
        ).astype(far_mask_dtype, copy=False)

    # Match the actual integer raster in each axis. Converting through the
    # COLMAP convention preserves the pixel-center (+/- 0.5) resize rule.
    sx = out_width / input_width
    sy = resized_height / input_height
    camera_intrinsics = opencv_to_colmap_intrinsics(camera_intrinsics)
    camera_intrinsics[0, :] *= sx
    camera_intrinsics[1, :] *= sy
    camera_intrinsics = colmap_to_opencv_intrinsics(camera_intrinsics)

    if min_vertical_pad_px > 0:
        max_content_height = out_height - 2 * min_vertical_pad_px
        if resized_height > max_content_height:
            margin_y = resized_height - max_content_height
            principal_fraction_y = float(
                np.clip(camera_intrinsics[1, 2] / max(resized_height, 1), 0.0, 1.0)
            )
            top = int(np.clip(np.rint(principal_fraction_y * margin_y), 0, margin_y))
            bottom = top + max_content_height
            crop_bbox = (0, top, out_width, bottom)
            image = image.crop(crop_bbox)
            if depthmap is not None:
                depthmap = depthmap[top:bottom]
            if normal is not None:
                normal = normal[top:bottom]
            if far_mask is not None:
                far_mask = far_mask[top:bottom]
            camera_intrinsics = camera_intrinsics.copy()
            camera_intrinsics[1, 2] -= top
            resized_height = max_content_height

    if resized_height > out_height:
        top = int(np.round((resized_height - out_height) * 0.5))
        top = min(max(top, 0), resized_height - out_height)
        crop_bbox = (0, top, out_width, top + out_height)
        image = image.crop(crop_bbox)
        if depthmap is not None:
            depthmap = depthmap[top : top + out_height]
        if normal is not None:
            normal = normal[top : top + out_height]
        if far_mask is not None:
            far_mask = far_mask[top : top + out_height]
        camera_intrinsics = camera_intrinsics.copy()
        camera_intrinsics[1, 2] -= top
        content_bbox = (0, 0, out_width, out_height)
    elif resized_height < out_height:
        top = (out_height - resized_height) // 2
        bottom = out_height - resized_height - top

        fill_u8 = tuple(int(np.clip(np.round(v * 255.0), 0, 255)) for v in pad_rgb)
        padded_images = []
        for resized_image in image.images:
            canvas = PIL.Image.new(resized_image.mode, (out_width, out_height), fill_u8)
            canvas.paste(resized_image, (0, top))
            padded_images.append(canvas)
        image = ImageList(padded_images)

        def _pad_array(array, fill_value=0):
            if array is None:
                return None
            shape = (out_height, out_width) + tuple(array.shape[2:])
            canvas = np.full(shape, fill_value, dtype=array.dtype)
            canvas[top : top + resized_height] = array
            return canvas

        depthmap = _pad_array(depthmap, 0)
        normal = _pad_array(normal, 0)
        far_mask = _pad_array(far_mask, False)
        camera_intrinsics = camera_intrinsics.copy()
        camera_intrinsics[1, 2] += top
        content_bbox = (0, top, out_width, top + resized_height)
        assert bottom >= 0
    else:
        content_bbox = (0, 0, out_width, out_height)

    return (
        image.to_pil(),
        depthmap,
        camera_intrinsics,
        normal,
        far_mask,
        content_bbox,
    )


def _normalize_rgb_pad_value(value):
    if value is None:
        vals = [0.485, 0.456, 0.406]
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            vals = [float(x.strip()) for x in text[1:-1].split(",") if x.strip()]
        elif "," in text:
            vals = [float(x.strip()) for x in text.split(",") if x.strip()]
        else:
            vals = [float(text)]
    else:
        try:
            vals = [float(x) for x in value]
        except TypeError:
            vals = [float(value)]
    if len(vals) == 1:
        vals = vals * 3
    if len(vals) != 3:
        raise ValueError(f"pad_value must have 1 or 3 values, got {value!r}")
    if any((v < 0.0 or v > 1.0 or not np.isfinite(v)) for v in vals):
        raise ValueError(f"pad_value must be finite and in [0,1], got {value!r}")
    return tuple(float(v) for v in vals)


def _pil_fill_for_pad_value(mode, pad_value):
    vals = tuple(int(round(v * 255.0)) for v in _normalize_rgb_pad_value(pad_value))
    if mode == "RGBA":
        return vals + (255,)
    if mode == "L":
        return int(round(sum(vals) / 3.0))
    return vals


def _pad_array_to_size(arr, output_resolution, fill_value=0):
    if arr is None:
        return None
    out_w, out_h = int(output_resolution[0]), int(output_resolution[1])
    h, w = arr.shape[:2]
    top = max((out_h - h) // 2, 0)
    left = max((out_w - w) // 2, 0)
    if arr.ndim == 2:
        canvas = np.full((out_h, out_w), fill_value, dtype=arr.dtype)
        canvas[top:top + h, left:left + w] = arr
    else:
        canvas = np.full((out_h, out_w, arr.shape[2]), fill_value, dtype=arr.dtype)
        canvas[top:top + h, left:left + w, :] = arr
    return canvas


def rescale_image_depthmap_long_edge_pad_crop(
    image,
    depthmap,
    camera_intrinsics,
    output_resolution,
    pad_value=(0.485, 0.456, 0.406),
    normal=None,
    far_mask=None,
):
    """Resize by matching long edges, then center-crop overflow or mean-pad gaps."""
    image = ImageList(image)
    input_resolution = np.array(image.size, dtype=np.float32)  # (W,H)
    output_resolution = np.array(output_resolution, dtype=np.int64)
    assert output_resolution.shape == (2,)
    if depthmap is not None:
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    src_long = max(float(input_resolution[0]), float(input_resolution[1]), 1.0)
    tgt_long = max(float(output_resolution[0]), float(output_resolution[1]), 1.0)
    scale_final = tgt_long / src_long
    resized_resolution = np.maximum(
        np.floor(input_resolution * scale_final + 1e-6).astype(np.int64),
        1,
    )

    image = image.resize(tuple(resized_resolution), resample=lanczos if scale_final < 1 else bicubic)
    if depthmap is not None:
        depthmap = cv2.resize(depthmap, tuple(resized_resolution), fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)
    if normal is not None:
        normal = cv2.resize(normal, tuple(resized_resolution), fx=scale_final,
                            fy=scale_final, interpolation=cv2.INTER_NEAREST)
    if far_mask is not None:
        far_mask = cv2.resize(far_mask, tuple(resized_resolution), fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)

    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, resized_resolution, scaling=scale_final
    )

    # Center-crop any overflow before padding missing short-edge area.
    margins = resized_resolution - output_resolution
    crop_l = int(max(margins[0] // 2, 0))
    crop_t = int(max(margins[1] // 2, 0))
    crop_r = int(crop_l + min(resized_resolution[0], output_resolution[0]))
    crop_b = int(crop_t + min(resized_resolution[1], output_resolution[1]))
    if crop_l or crop_t or crop_r < resized_resolution[0] or crop_b < resized_resolution[1]:
        image, depthmap, camera_intrinsics, normal, far_mask = crop_image_depthmap(
            image.to_pil(), depthmap, camera_intrinsics, (crop_l, crop_t, crop_r, crop_b),
            normal=normal, far_mask=far_mask
        )
        image = ImageList(image)

    pad_w = int(output_resolution[0] - image.size[0])
    pad_h = int(output_resolution[1] - image.size[1])
    if pad_w > 0 or pad_h > 0:
        left = max(pad_w // 2, 0)
        top = max(pad_h // 2, 0)
        pad_mask = np.ones((int(output_resolution[1]), int(output_resolution[0])), dtype=bool)
        pad_mask[top:top + image.size[1], left:left + image.size[0]] = False
        canvas = PIL.Image.new(
            image.images[0].mode,
            tuple(int(x) for x in output_resolution),
            color=_pil_fill_for_pad_value(image.images[0].mode, pad_value),
        )
        canvas.paste(image.images[0], (left, top))
        canvas.info["_streampi3_mean_padding_mask"] = pad_mask
        canvas.info["_streampi3_mean_padding_value"] = _normalize_rgb_pad_value(pad_value)
        image = ImageList(canvas)
        depthmap = _pad_array_to_size(depthmap, output_resolution, fill_value=0)
        normal = _pad_array_to_size(normal, output_resolution, fill_value=0)
        far_mask = _pad_array_to_size(far_mask, output_resolution, fill_value=0)
        camera_intrinsics = camera_intrinsics.copy()
        camera_intrinsics[0, 2] += left
        camera_intrinsics[1, 2] += top

    return image.to_pil(), depthmap, camera_intrinsics, normal, far_mask

def center_crop_image_depthmap(image, depthmap, camera_intrinsics, crop_scale, normal=None, far_mask=None):
    """
    Jointly center-crop an image and its depthmap, and adjust the camera intrinsics accordingly.

    Parameters:
    - image: PIL.Image or similar, the input image.
    - depthmap: np.ndarray, the corresponding depth map.
    - camera_intrinsics: np.ndarray, the 3x3 camera intrinsics matrix.
    - crop_scale: float between 0 and 1, the fraction of the image to keep.

    Returns:
    - cropped_image: PIL.Image, the center-cropped image.
    - cropped_depthmap: np.ndarray, the center-cropped depth map.
    - adjusted_intrinsics: np.ndarray, the adjusted camera intrinsics matrix.
    """
    # Ensure crop_scale is valid
    assert 0 < crop_scale <= 1, "crop_scale must be between 0 and 1"

    # Convert image to ImageList for consistent processing
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (width, height)
    if depthmap is not None:
        # Ensure depthmap matches the image size
        assert depthmap.shape[:2] == tuple(image.size[::-1]), "Depthmap size must match image size"

    # Compute output resolution after cropping
    output_resolution = np.floor(input_resolution * crop_scale).astype(int)
    # get the correct crop_scale
    crop_scale = output_resolution / input_resolution

    # Compute margins (amount to crop from each side)
    margins = input_resolution - output_resolution
    offset = margins / 2  # Since we are center cropping

    # Calculate the crop bounding box
    left, top = offset.astype(int)
    right = left + output_resolution[0]
    bottom = top + output_resolution[1]
    crop_bbox = (left, top, right, bottom)

    # Crop the image and depthmap
    image = image.crop(crop_bbox)
    if depthmap is not None:
        depthmap = depthmap[top:bottom, left:right]
    if normal is not None:
        normal = normal[top:bottom, left:right]
    if far_mask is not None:
        far_mask = far_mask[top:bottom, left:right]

    # Adjust the camera intrinsics
    adjusted_intrinsics = camera_intrinsics.copy()

    # Adjust focal lengths (fx, fy)                         # no need to adjust focal lengths for cropping
    # adjusted_intrinsics[0, 0] /= crop_scale[0]  # fx
    # adjusted_intrinsics[1, 1] /= crop_scale[1]  # fy

    # Adjust principal point (cx, cy)
    adjusted_intrinsics[0, 2] -= left  # cx
    adjusted_intrinsics[1, 2] -= top  # cy

    return image.to_pil(), depthmap, adjusted_intrinsics, normal, far_mask


def camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution, scaling=1, offset_factor=0.5, offset=None):
    # Margins to offset the origin
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix


def crop_image_depthmap(image, depthmap, camera_intrinsics, crop_bbox, normal=None, far_mask=None):
    """
    Return a crop of the input view.
    """
    image = ImageList(image)
    left, top, right, bottom = crop_bbox

    image = image.crop((left, top, right, bottom))
    depthmap = depthmap[top:bottom, left:right]
    if normal is not None:
        normal = normal[top:bottom, left:right]
    if far_mask is not None:
        far_mask = far_mask[top:bottom, left:right]

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= left
    camera_intrinsics[1, 2] -= top

    return image.to_pil(), depthmap, camera_intrinsics, normal, far_mask


def bbox_from_intrinsics_in_out(input_camera_matrix, output_camera_matrix, output_resolution):
    out_width, out_height = output_resolution
    left, top = np.int32(np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2]))
    crop_bbox = (left, top, left + out_width, top + out_height)
    return crop_bbox
