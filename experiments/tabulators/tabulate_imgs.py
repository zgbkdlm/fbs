def to_imsave(img):
    img = normalise(img, method='clip')
    return img[..., 0] if nchannels == 1 else img
