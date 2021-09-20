import io

from PIL import Image


def compress_img_PIL(bytes, compress_rate=0.2):
    '''
    img.resize() 方法可以缩小可以放大
    img.thumbnail() 方法只能缩小
    :param way:
    :param compress_rate:
    :param show:
    :return:
    '''
    img = Image.open(bytes)
    w, h = img.size
    # 方法一：使用resize改变图片分辨率，但是图片内容并不丢失，不是裁剪
    img_resize = img.resize((int(w * compress_rate), int(h * compress_rate)))
    resize_w, resieze_h = img_resize.size
    img_bytes = io.BytesIO()
    img_resize.save(img_bytes, format='png')
    img_bytes = img_bytes.getvalue()
    return io.BytesIO(img_bytes)


