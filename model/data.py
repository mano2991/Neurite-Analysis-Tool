import glob
import os

import tensorflow as tf


def enforce_4D(image, dtype=tf.float32):
    """Enforce an image shape [batch, x, y, channels]"""
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image)
    ndims = image.get_shape().ndims
    if ndims == 2:
        image = image[None, :, :, None]
    elif ndims == 3 and image.shape[0] == 1:
        image = image[:, :, :, None]
    elif ndims == 3:
        image = image[None, :, :, :]
    elif ndims != 4:
        raise ValueError('Unknown image shape: {}'.format(image.shape))
    return tf.cast(image, dtype)


def clipped_rescale(image, min_val=0, max_val=1, dtype=tf.float32):
    """Clip image values to [min_val, max_val] and rescale to [0, 1]"""
    image = tf.cast(image, dtype)
    image = tf.clip_by_value(image, min_val, max_val)
    min_val = tf.maximum(tf.cast(min_val, dtype), tf.reduce_min(image))
    max_val = tf.minimum(tf.cast(max_val, dtype), tf.reduce_max(image))
    image = (image - min_val) / (max_val - min_val)
    return image


def wtko_png_loader(path):
    """Load image/label for a given path - returns both WT/C-KO and F/M labels"""
    label = tf.cast(tf.strings.regex_full_match(path, '(.*)(KO_)(.+)'), tf.float32)
    label2 = tf.cast(tf.strings.regex_full_match(path, '(.*)(_M_)(.+)'), tf.float32)
    label = tf.stack([label, label2], axis=0)
    image = tf.image.decode_png(tf.io.read_file(path), channels=1)
    image = tf.cast(image, tf.float32)
    min_pix = tf.reduce_min(image)
    max_pix = tf.reduce_max(image)
    image = (image - min_pix) / (max_pix - min_pix)
    return image, label


def replate_png_loader(path):
    """Load image/label for a given path - specific to WT vs WT-Replate data"""
    label = tf.cast(tf.strings.regex_full_match(path, '(.+)(Replate)(.+)'), tf.float32)
    label = tf.ones([1], dtype=tf.float32) * label
    image = tf.image.decode_png(tf.io.read_file(path), channels=1)
    image = tf.cast(image, tf.float32)
    min_pix = tf.reduce_min(image)
    max_pix = tf.reduce_max(image)
    image = (image - min_pix) / (max_pix - min_pix)
    return image, label


def pattern_png_loader(path):
    """Load image/label for a given path - specific to Pattern vs Unpatterned data"""
    label = tf.cast(tf.strings.regex_full_match(path, '(.*)( KO )(.+)'), tf.float32)
    label2 = tf.cast(tf.strings.regex_full_match(path, '(.*)(M )(.+)'), tf.float32)
    label3 = tf.cast(tf.strings.regex_full_match(path, '(.*)(Blank)(.+)'), tf.float32)
    label = tf.stack([label, label2, label3], axis=0)
    image = tf.image.decode_png(tf.io.read_file(path), channels=1)
    image = clipped_rescale(image, 0, 1, dtype=tf.float32)
    return image, label


def convert_to_sobel(image, label):
    """Apply a sobel edge detection to the image"""
    out_shape = tf.shape(image)
    image = tf.reshape(image, [-1, 1024, 1360, 1])  # the size of images in our dataset
    sobels = tf.image.sobel_edges(image)
    sobels = tf.reduce_sum(sobels ** 2, axis=-1)
    sobels = tf.sqrt(sobels)

    # rescale back to [0, 1]
    min_pix = tf.reduce_min(sobels, axis=(1, 2, 3), keepdims=True)
    max_pix = tf.reduce_max(sobels, axis=(1, 2, 3), keepdims=True)
    sobels = tf.divide(tf.subtract(sobels, min_pix), tf.subtract(max_pix, min_pix))
    sobels = tf.reshape(sobels, out_shape)
    return sobels, label


def get_augmenter(random_crop=[30, 30], flip_h=True, flip_v=True):
    """Return an augmentation function

    Parameters
    ----------
    random_crop : [int, int]
        The maximum number of pixels to randomly crop from the image edges
    flip_h : bool
        Whether to allow random horizontal flipping
    flip_v : bool
        Whether to allow random vertical flipping
    """
    def augmenter(image, label):
        image_size = tf.shape(image)
        image = tf.image.resize_with_crop_or_pad(image, image_size[0] + random_crop[0], image_size[1] + random_crop[1])
        image = tf.image.random_crop(image, image_size)
        if flip_h:
            image = tf.image.random_flip_left_right(image)
        if flip_v:
            image = tf.image.random_flip_up_down(image)
        return image, label
    return augmenter


def prepare_dataset(base_image_dir='TrainImages', batch_size=32, is_replate=False, is_training=True, cache_location=None):
    """Prepare the training/testing image dataset for WT/C-KO or WT/WT-Replate classification

    Parameters
    ----------
    base_image_dir : os.PathLike
        the parent directory of the KO/WT images
    batch_size : int
        the size of the batches in the dataset (the default is 32)
    is_replate : bool
        If True, loads the data as if it is from the WT vs WT-Replate dataset. If False, loads as if it is from the WT vs C-KO dataset.
    is_training : bool
        Whether the data is the training dataset and should have augmentations applied
    cache_location : os.PathLike, optional
        the location to cache the dataset to
    """
    image_paths = glob.glob(os.path.join(base_image_dir, '*.png'))
    SHUFFLE_SIZE = len(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    image_loader = replate_png_loader if is_replate else wtko_png_loader
    dataset = dataset.map(image_loader, num_parallel_calls=-1)
    dataset = dataset.map(convert_to_sobel, num_parallel_calls=-1)
    if cache_location is not None:
        dataset = dataset.cache(cache_location)

    if is_training:
        dataset = dataset.shuffle(buffer_size=SHUFFLE_SIZE)
        augment_func = get_augmenter()
        dataset = dataset.map(augment_func, num_parallel_calls=-1)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=-1)
    return dataset


def prepare_pattern_dataset(base_image_dir="TrainImagesPattern", batch_size=32, is_training=True, cache_location=None):
    """Prepare the training/testing image dataset for Pattern/Unpatterned classification

    Parameters
    ----------
    base_image_dir : str, optional
        The parent directory of the Pattern/Unpatterned images
    batch_size : int, optional
        the size of the batches in the dataset (the default is 32)
    is_training : bool, optional
        Whether the data is the training dataset and should have augmentations applied
    cache_location : os.PathLike, optional
        the location to cache the dataset to
    """
    image_paths = glob.glob(os.path.join(base_image_dir, '*.png'))
    SHUFFLE_SIZE = len(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    image_loader = pattern_png_loader
    dataset = dataset.map(image_loader, num_parallel_calls=-1)
    dataset = dataset.map(convert_to_sobel, num_parallel_calls=-1)
    if cache_location is not None:
        dataset = dataset.cache(cache_location)
    if is_training:
        dataset = dataset.shuffle(buffer_size=SHUFFLE_SIZE)
        augment_func = get_augmenter()
        dataset = dataset.map(augment_func, num_parallel_calls=-1)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=-1)
    return dataset