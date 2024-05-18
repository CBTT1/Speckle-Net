from PIL import Image, ImageOps


def keep_image_size_open(path, size=(64, 64), reverse=False):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)

    if reverse:
        inverted_image = ImageOps.invert(mask)
        return inverted_image
    else:
        return mask


