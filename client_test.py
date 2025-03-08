from server.instance import executor_instances, ExecutorInstance
from server.request_extraction import to_pil_image
from manga_translator import Config
import json
import glob

executor_instances.register(ExecutorInstance(ip="127.0.0.1", port=5004))

cfg = {
    "detector": {
        "detector": "default",
        "detection_size": 2048,
        "text_threshold": 0.5,
        "box_threshold": 0.7,
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
        "translator": "sakura",
        "target_lang": "CHS"
    },
    "colorizer": {
        "colorizer": "none",
        "colorization_size": 576,
        "denoise_sigma": 30
    },
    "inpainter": {
        "inpainter": "lama_large",
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
import glob
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import pathlib
import argparse


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


async def process_image(image_path: str, output_dir: str, config: Config, ctor: ExecutorInstance, quality: int = 90):
    """
    处理单个图片的函数
    
    参数:
    image_path: 输入图片路径
    output_dir: 输出目录
    config: 配置对象
    ctor: 执行器实例
    quality: 输出图片质量
    """
    try:
        # 创建输出文件夹
        base_name = os.path.basename(image_path).split('.')[0]
        image_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # 读取图片
        with open(image_path, 'rb') as f:
            img = await to_pil_image(f.read())
        
        # OCR识别
        result = await ctor.sent_ocr(img, config)
        
        # 翻译
        result = await ctor.sent_translate_ctx(config, result)
        
        if isinstance(result['result'], list):
            # 保存区域并获取元数据文件路径
            metadata_path = save_regions_to_png(result['result'], image_output_dir, base_filename=f"{base_name}_region")
            
            # 合并图片
            final = merge_regions_to_image(result['img_inpainted'], metadata_path)
            
            # 保存最终结果
            output_path = os.path.join(output_dir, f"{base_name}_translated.webp")
            save_as_webp(final, output_path, quality)
        else:
            # 当result['result']为WebPImageFile时，直接保存
            output_path = os.path.join(output_dir, f"{base_name}_translated.webp")
            result['result'].save(output_path, quality=quality)
            print(f"图像已保存到: {output_path}")
        
        return output_path
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        raise e

async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量处理漫画图片翻译')
    parser.add_argument('--input_dir', type=str, default='/home/yxa/chatglm_demo/manga',
                        help='输入图片目录路径')
    parser.add_argument('--output_dir', type=str, default='/home/yxa/chatglm_demo/manga-image-translator/result/batch_result',
                        help='输出图片目录路径')
    parser.add_argument('--max_workers', type=int, default=1,
                        help='并发处理的最大线程数')
    parser.add_argument('--quality', type=int, default=90,
                        help='输出webp图片的质量，范围0-100')
    
    args = parser.parse_args()
    
    # 使用命令行参数
    input_dir = args.input_dir
    output_dir = args.output_dir
    max_workers = args.max_workers
    quality = args.quality
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"最大线程数: {max_workers}")
    print(f"输出图片质量: {quality}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有webp图片
    image_paths = glob.glob(os.path.join(input_dir, "*.webp"))
    
    if not image_paths:
        print(f"在目录 {input_dir} 中没有找到webp图片")
        return
    
    print(f"找到 {len(image_paths)} 个webp图片")
    
    # 初始化配置和执行器
    config = Config.parse_raw(json.dumps(cfg))
    ctor = await executor_instances.find_executor()
    
    # 使用信号量限制并发任务数
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_with_semaphore(image_path):
        async with semaphore:
            return await process_image(image_path, output_dir, config, ctor, quality)
    
    # 创建任务列表
    tasks = [asyncio.create_task(process_with_semaphore(image_path)) for image_path in image_paths]
    
    # 使用tqdm显示进度
    results = []
    errors = []
    
    with tqdm(total=len(tasks), desc="处理图片") as pbar:
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
            except Exception as e:
                errors.append(str(e))
            finally:
                pbar.update(1)
    
    # 显示处理结果
    success_count = len(results)
    failed_count = len(errors)
    print(f"处理完成: 成功 {success_count} 张, 失败 {failed_count} 张")
    
    if failed_count > 0:
        print("失败列表:")
        for error in errors[:5]:  # 只显示前5个错误
            print(f"- {error}")
        if len(errors) > 5:
            print(f"... 以及其他 {len(errors) - 5} 个错误")
    
    print(f"翻译结果保存在: {output_dir}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
