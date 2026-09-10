"""
把任意图片处理成 200x200 的 JPG。
两种模式（通过 MODE 切换）：
  - "fit"   : 等比缩放到最长边=200，居中放置（不足处填白边，不变形）
  - "crop"  : 等比缩放到最短边=200，再中心裁剪成正好 200x200（可能丢部分内容）

依赖：pip install pillow
用法：python pictures\Picture_compression.py 输入图片 输出图片
"""

from PIL import Image, ImageOps
import sys

MODE = "crop"          # 可选 "fit"  /  "crop"
SIZE = 200             # 目标边长
QUALITY = 85           # 输出 JPG 质量


def make_200(src_path, dst_path):
    with Image.open(src_path) as im:
        im = ImageOps.exif_transpose(im)          # 修正手机照片的旋转
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")

        w, h = im.size

        if MODE == "crop":
            # 等比缩放：让短边刚好等于 200，再中心裁剪
            scale = SIZE / min(w, h)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
            left = (new_w - SIZE) // 2
            top = (new_h - SIZE) // 2
            im = im.crop((left, top, left + SIZE, top + SIZE))

        else:  # "fit"
            # 等比缩放：让长边不超过 200，居中贴到 200x200 白底上
            im.thumbnail((SIZE, SIZE), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
            left = (SIZE - im.width) // 2
            top = (SIZE - im.height) // 2
            canvas.paste(im, (left, top))
            im = canvas

        im.save(dst_path, "JPEG", quality=QUALITY, optimize=True)
        print(f"{src_path}  ({w}x{h}) -> {dst_path}  ({im.width}x{im.height})")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    dst = sys.argv[2] if len(sys.argv) > 2 else "output_200x200.jpg"
    make_200(src, dst)
