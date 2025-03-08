from server.instance import executor_instances, ExecutorInstance
from server.request_extraction import to_pil_image
from manga_translator import Config
import json

executor_instances.register(ExecutorInstance(ip="127.0.0.1", port=5004))

cfg = {
    "detector": {
        "detector": "ctd",
        "detection_size": 2048,
        "text_threshold": 0.2,
        "box_threshold": 0.3,
        "unclip_ratio": 2.3,
        "det_rotate": False,
        "det_auto_rotate": False,
        "det_invert": False,
        "det_gamma_correct": False
    },
    "render": {
        "renderer": "default",
        "alignment": "auto",
        "disable_font_border": False,
        "font_size_offset": 0,
        "direction": "auto",
        "uppercase": False,
        "lowercase": False,
        "no_hyphenation": False,
        "return_region_only": True
    },
    "upscale": {
        "upscaler": "esrgan",
        "revert_upscaling": False,
        "upscale_ratio": None
    },
    "translator": {
        "translator":"original"
        # "translator": "gpt3.5",
        # "target_lang": "CHS"
    },
    "colorizer": {
        "colorizer": "none",
        "colorization_size": 576,
        "denoise_sigma": 30
    },
    "inpainter": {
        "inpainter": "sd",
        "inpainting_size": 2048,
        "inpainting_precision": "fp16"
    },
    "ocr": {
        "ocr": "48px",
        "min_text_length": 0,
        "ignore_bubble": 0,
        "use_mocr_merge": False
    },
    "kernel_size": 3,
    "mask_dilation_offset": 0
}



import os
import cv2
import numpy as np
from typing import List, Union, Dict, Any


def save_regions_to_png(regions: List[Dict[str, Any]], output_dir: str, base_filename: str = "region") -> str:
    """
    保存渲染区域为PNG文件并存储相关元数据
    参数:
    regions: 由dispatch函数返回的区域列表
    output_dir: 输出目录路径
    base_filename: 基础文件名前缀
    返回:
    元数据文件的路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 创建元数据列表
    metadata = []
    # 为每个区域保存图像和元数据
    for i, region in enumerate(regions):
        # 准备文件名
        image_filename = f"{base_filename}{i}.png"
        image_path = os.path.join(output_dir, image_filename)
        # 保存图像
        cv2.imwrite(image_path, cv2.cvtColor(region['image'], cv2.COLOR_RGB2BGR))
        # 保存与图像相关的元数据
        region_metadata = {
            'image_filename': image_filename,
            'position': region['position'],
            'size': region['size'],
            # 'points': region['points'],
            'text': region['text']
        }
        metadata.append(region_metadata)
# 将元数据保存为JSON文件
    metadata_filename = f"{base_filename}metadata.json"
    metadata_path = os.path.join(output_dir, metadata_filename)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(regions)} 个区域到 {output_dir}")
    return metadata_path

def merge_regions_to_image(original_image: np.ndarray, metadata_path: str) -> np.ndarray:
    """
    从PNG文件读取区域图像并合并到原始图像中
    参数:
    original_image: 原始图像
    metadata_path: 元数据文件路径
    返回:
    合并后的图像
    """
    # 创建结果图像的副本
    result_image = original_image.copy()
    # 读取元数据
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    # 获取元数据文件所在目录
    base_dir = os.path.dirname(metadata_path)
    # 处理每个区域
    for region_meta in metadata:
        # 获取区域图像路径
        image_path = os.path.join(base_dir, region_meta['image_filename'])
        # 读取区域图像
        region_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        if region_image is None:
            print(f"无法读取区域图像: {image_path}")
            continue
        # 获取位置和大小
        x, y = region_meta['position']
        w, h = region_meta['size']
        # 检查图像尺寸是否匹配
        # if region_image.shape[:2] != (h, w):
        #     logger.warning(f"区域图像尺寸不匹配: {image_path}，调整大小")
        #     region_image = cv2.resize(region_image, (w, h))
        # 将区域图像放回原始图像
        result_image[y:y+h, x:x+w] = region_image
    print(f"已合并 {len(metadata)} 个区域到原始图像")
    return result_image

def save_as_webp(image: np.ndarray, output_path: str, quality: int = 80):
    """
    将numpy数组格式的图像保存为webp格式
    参数:
    image: numpy数组格式的图像
    output_path: 输出文件路径
    quality: 压缩质量,范围0-100,默认80
    """
    # 确保图像是BGR格式
    # if len(image.shape) == 3 and image.shape[2] == 3:
    #     image_bgr = image
    # else:
    #     image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 编码为webp格式
    _, encoded_image = cv2.imencode('.webp', image_bgr, 
                                  [cv2.IMWRITE_WEBP_QUALITY, quality])
    
    # 写入文件
    encoded_image.tofile(output_path)
    print(f"图像已保存到: {output_path}")


async def main():
    with open('/home/yxa/chatglm_demo/000000001.webp', 'rb') as f:
        img = await to_pil_image(f.read())
    config = Config.parse_raw(json.dumps(cfg))
    ctor = await executor_instances.find_executor()
    result = await ctor.sent_ocr(img, config)
    print("ocr finish")
    result = await ctor.sent_translate_ctx(config, result)
    print("inpaint finish")
    save_regions_to_png(result['result'], "/home/yxa/chatglm_demo/manga-image-translator/result/result")
    final = merge_regions_to_image(result['img_inpainted'], '/home/yxa/chatglm_demo/manga-image-translator/result/result/regionmetadata.json')
    save_as_webp(final, '/home/yxa/chatglm_demo/manga-image-translator/result/result/final.webp', 90)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
