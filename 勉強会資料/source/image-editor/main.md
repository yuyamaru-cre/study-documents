```
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
import uuid
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import asyncio
from pathlib import Path
import aiofiles
import zipfile
import io
import xml.etree.ElementTree as ET
import re
from text_image_replacer import TextImageReplacer, create_template_config

app = FastAPI(title="Image Editor API", version="1.0.0")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# アップロードディレクトリの設定
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# 静的ファイル配信の設定
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")


class ImageResponse(BaseModel):
    url: str
    filename: str


class ImageBatchResponse(BaseModel):
    original_url: str
    processed_url: str
    original_filename: str
    processed_filename: str
    status: str
    error_message: Optional[str] = None


class OperationRequest(BaseModel):
    id: str
    operation: str
    parameters: Dict[str, float]
    enableCrop: Optional[bool] = None


class ProcessBatchRequest(BaseModel):
    image_urls: List[str]
    operations: List[OperationRequest]
    output_format: Optional[str] = "original"


@app.get("/")
async def root():
    return {"message": "Image Editor API", "version": "1.0.0"}


@app.post("/api/upload", response_model=ImageResponse)
async def upload_image(file: UploadFile = File(...)):
    """画像ファイルをアップロードする"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload an image."
        )

    # ファイル名を生成
    file_extension = file.filename.split(".")[-1] if file.filename else "jpg"
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = UPLOAD_DIR / unique_filename

    # ファイルを保存
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    return ImageResponse(url=f"/uploads/{unique_filename}", filename=unique_filename)


@app.post("/api/upload-batch", response_model=List[ImageResponse])
async def upload_batch_images(files: List[UploadFile] = File(...)):
    """複数の画像ファイルを一括アップロードする"""
    uploaded_files = []

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Please upload images only.",
            )

        # ファイル名を生成
        file_extension = file.filename.split(".")[-1] if file.filename else "jpg"
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = UPLOAD_DIR / unique_filename

        # ファイルを保存
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        uploaded_files.append(
            ImageResponse(url=f"/uploads/{unique_filename}", filename=unique_filename)
        )

    return uploaded_files


@app.post("/api/process-batch", response_model=List[ImageBatchResponse])
async def process_batch_images(request: ProcessBatchRequest):
    """複数の画像を一括処理する"""
    results = []

    for image_url in request.image_urls:
        try:
            # アップロード画像のパスを取得
            if image_url.startswith("/uploads/") or "/uploads/" in image_url:
                filename = image_url.split("/")[-1]
                input_path = UPLOAD_DIR / filename
            else:
                results.append(
                    ImageBatchResponse(
                        original_url=image_url,
                        processed_url="",
                        original_filename=filename,
                        processed_filename="",
                        status="error",
                        error_message="Invalid image URL",
                    )
                )
                continue

            if not input_path.exists():
                results.append(
                    ImageBatchResponse(
                        original_url=image_url,
                        processed_url="",
                        original_filename=filename,
                        processed_filename="",
                        status="error",
                        error_message="Image not found",
                    )
                )
                continue

            # 出力ファイル名を生成（元の拡張子を保持または指定された形式に変換）
            original_extension = filename.split(".")[-1].lower()

            if request.output_format == "original":
                output_extension = original_extension
            elif request.output_format == "png":
                output_extension = "png"
            elif request.output_format == "jpg":
                output_extension = "jpg"
            elif request.output_format == "webp":
                output_extension = "webp"
            else:
                output_extension = original_extension

            output_filename = f"processed_{uuid.uuid4()}.{output_extension}"
            output_path = PROCESSED_DIR / output_filename

            # 複数の画像処理を順番に実行
            await process_multiple_operations_async(
                input_path, output_path, request.operations
            )

            results.append(
                ImageBatchResponse(
                    original_url=image_url,
                    processed_url=f"/processed/{output_filename}",
                    original_filename=filename,
                    processed_filename=output_filename,
                    status="success",
                    error_message=None,
                )
            )

        except Exception as e:
            results.append(
                ImageBatchResponse(
                    original_url=image_url,
                    processed_url="",
                    original_filename=filename,
                    processed_filename="",
                    status="error",
                    error_message=str(e),
                )
            )

    return results


async def process_multiple_operations_async(
    input_path: Path, output_path: Path, operations: List[OperationRequest]
):
    """複数の操作を順番に処理する（PILのみ使用、メモリ内で処理して品質劣化を最小化）"""
    if not operations:
        raise ValueError("操作が指定されていません")

    def process_sync():
        # PILで画像を読み込み（カラープロファイル保持）
        with Image.open(str(input_path)) as original_image:
            # 画像をコピーして処理用にする
            image = original_image.copy()

            # DPI情報を保持
            dpi_value = None
            for operation in operations:
                if operation.operation == "dpi":
                    dpi_value = int(operation.parameters.get("dpi", 72))

            # 全ての操作を順番に適用
            for operation in operations:
                operation_params = operation.parameters.copy()
                if operation.operation == "resize" and operation.enableCrop is not None:
                    operation_params["crop"] = 1 if operation.enableCrop else 0

                image = apply_pil_operation(image, operation.operation, operation_params)

            # DPI情報がある場合は保存時に適用
            if dpi_value:
                # メタデータ保持保存（DPI情報付き）
                try:
                    # 元画像からメタデータを取得
                    exif = original_image.getexif()
                    icc_profile = original_image.info.get('icc_profile')

                    # 保存形式を決定
                    output_extension = str(output_path).lower()
                    if output_extension.endswith('.png'):
                        save_format = 'PNG'
                    elif output_extension.endswith(('.jpg', '.jpeg')):
                        save_format = 'JPEG'
                    elif output_extension.endswith('.webp'):
                        save_format = 'WEBP'
                    else:
                        save_format = 'JPEG'

                    save_kwargs = {
                        'format': save_format,
                        'dpi': (dpi_value, dpi_value),
                        'optimize': True
                    }

                    if save_format in ['JPEG', 'WEBP']:
                        save_kwargs['quality'] = 95
                    elif save_format == 'PNG':
                        save_kwargs['compress_level'] = 1

                    # EXIFデータがある場合は保持
                    if exif and save_format in ['JPEG', 'WEBP']:
                        save_kwargs['exif'] = exif

                    # ICCプロファイルがある場合は保持
                    if icc_profile:
                        save_kwargs['icc_profile'] = icc_profile

                    image.save(str(output_path), **save_kwargs)

                except Exception as e:
                    print(f"Warning: Could not save with DPI metadata: {e}")
                    preserve_image_metadata_and_save(input_path, output_path, image)
            else:
                # 通常の保存（メタデータ保持）
                preserve_image_metadata_and_save(input_path, output_path, image)

    # CPU集約的な処理を別スレッドで実行
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, process_sync)


def apply_pil_operation(image: Image.Image, operation: str, parameters: Dict[str, float]) -> Image.Image:
    """PILによる単一の画像処理操作を適用する（メモリ内処理）"""
    if operation == "resize":
        width = int(parameters.get("width", 800))
        height = int(parameters.get("height", 600))
        crop = int(parameters.get("crop", 0))  # 0=通常リサイズ, 1=トリミング付きリサイズ

        if crop == 1:
            # トリミング付きリサイズ：アスペクト比を維持してトリミング
            # ImageOps.fitを使用してスマートトリミング
            image = ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS)
        else:
            # 通常のリサイズ：アスペクト比を無視して指定サイズに変更
            image = image.resize((width, height), Image.Resampling.LANCZOS)

    elif operation == "rotate":
        angle = parameters.get("angle", 0)
        # 高品質な回転処理
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    elif operation == "blur":
        sigma = parameters.get("sigma", 1.0)
        # ガウシアンブラーを適用
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

    elif operation == "grayscale":
        # グレースケール変換（Lモードに変換してからRGBに戻す）
        grayscale = image.convert('L')
        image = grayscale.convert('RGB')

    elif operation == "sepia":
        # セピア効果をPILで実装
        # まずグレースケールに変換
        grayscale = image.convert('L')

        # セピア色調を作成
        sepia = Image.new('RGB', image.size)
        sepia_pixels = []

        for pixel in grayscale.getdata():
            # セピア色計算
            r = min(255, int(pixel * 1.0))
            g = min(255, int(pixel * 0.8))
            b = min(255, int(pixel * 0.6))
            sepia_pixels.append((r, g, b))

        sepia.putdata(sepia_pixels)
        image = sepia

    elif operation == "overlay":
        # 透過黒オーバーレイを適用
        opacity = parameters.get("opacity", 0.3)  # デフォルト30%

        # アルファ値を計算して黒いオーバーレイを作成
        alpha = int(opacity * 255)
        overlay_with_alpha = Image.new('RGBA', image.size, (0, 0, 0, alpha))

        # 元画像をRGBAに変換
        if image.mode != 'RGBA':
            image_rgba = image.convert('RGBA')
        else:
            image_rgba = image

        # 合成
        image = Image.alpha_composite(image_rgba, overlay_with_alpha)

        # 必要に応じてRGBに戻す
        if image.mode == 'RGBA':
            image = image.convert('RGB')



    elif operation == "dpi":
        # DPI変更（画像自体は変更せず、メタデータのみ変更）
        # DPI情報は保存時に設定されるため、ここでは何もしない
        pass

    return image


def preserve_image_metadata_and_save(original_path: Path, output_path: Path, processed_image: Image.Image):
    """元画像のメタデータ（EXIF、カラープロファイル）を保持して保存"""
    try:
        # 元画像を開いてメタデータを取得
        with Image.open(str(original_path)) as original:
            # EXIFデータを取得
            exif = original.getexif()

            # カラープロファイル（ICC Profile）を取得
            icc_profile = original.info.get('icc_profile')

            # その他のメタデータも取得
            info = original.info.copy()

            # 保存形式を決定
            output_extension = str(output_path).lower()
            if output_extension.endswith('.png'):
                save_format = 'PNG'
            elif output_extension.endswith(('.jpg', '.jpeg')):
                save_format = 'JPEG'
            elif output_extension.endswith('.webp'):
                save_format = 'WEBP'
            else:
                save_format = 'JPEG'

            # 保存オプション
            save_kwargs = {
                'format': save_format,
                'optimize': True
            }

            # 品質設定
            if save_format in ['JPEG', 'WEBP']:
                save_kwargs['quality'] = 95
            elif save_format == 'PNG':
                save_kwargs['compress_level'] = 1

            # EXIFデータがある場合は保持
            if exif and save_format in ['JPEG', 'WEBP']:
                save_kwargs['exif'] = exif

            # ICCプロファイルがある場合は保持
            if icc_profile:
                save_kwargs['icc_profile'] = icc_profile

            # DPI情報がある場合は保持
            if 'dpi' in info:
                save_kwargs['dpi'] = info['dpi']

            # 保存
            processed_image.save(str(output_path), **save_kwargs)
            print(f"Saved with metadata preservation: {output_path}")

    except Exception as e:
        print(f"Warning: Could not preserve metadata: {e}")
        # メタデータ保持に失敗した場合は通常の保存を行う
        if str(output_path).lower().endswith('.png'):
            processed_image.save(str(output_path), format='PNG', compress_level=1)
        elif str(output_path).lower().endswith(('.jpg', '.jpeg')):
            processed_image.save(str(output_path), format='JPEG', quality=95, optimize=True)
        elif str(output_path).lower().endswith('.webp'):
            processed_image.save(str(output_path), format='WEBP', quality=95, optimize=True)
        else:
            processed_image.save(str(output_path), format='JPEG', quality=95, optimize=True)


async def process_image_async(
    input_path: Path, output_path: Path, operation: str, parameters: Dict[str, float]
):
    """画像処理を非同期で実行（PILのみ使用）"""
    def process_sync():
        # PILで画像を読み込み（カラープロファイル保持）
        with Image.open(str(input_path)) as original_image:
            # 画像をコピーして処理用にする
            image = original_image.copy()

            print(f"Original image mode: {image.mode}")
            print(f"Original image size: {image.size}")

            # 操作を適用
            image = apply_pil_operation(image, operation, parameters)

            print(f"Processed image mode: {image.mode}")
            print(f"Processed image size: {image.size}")

            # DPI処理の場合は特別な保存処理
            if operation == "dpi":
                dpi = int(parameters.get("dpi", 72))
                try:
                    # 元画像からメタデータを取得
                    exif = original_image.getexif()
                    icc_profile = original_image.info.get('icc_profile')

                    # 保存形式を決定
                    output_extension = str(output_path).lower()
                    if output_extension.endswith('.png'):
                        save_format = 'PNG'
                    elif output_extension.endswith(('.jpg', '.jpeg')):
                        save_format = 'JPEG'
                    elif output_extension.endswith('.webp'):
                        save_format = 'WEBP'
                    else:
                        save_format = 'PNG'

                    save_kwargs = {
                        'format': save_format,
                        'dpi': (dpi, dpi),
                        'optimize': True
                    }

                    if save_format in ['JPEG', 'WEBP']:
                        save_kwargs['quality'] = 95
                    elif save_format == 'PNG':
                        save_kwargs['compress_level'] = 1

                    # EXIFデータがある場合は保持
                    if exif and save_format in ['JPEG', 'WEBP']:
                        save_kwargs['exif'] = exif

                    # ICCプロファイルがある場合は保持
                    if icc_profile:
                        save_kwargs['icc_profile'] = icc_profile

                    image.save(str(output_path), **save_kwargs)
                    print(f"Successfully saved with DPI {dpi}: {output_path}")

                except Exception as e:
                    print(f"Warning: Could not save with DPI metadata: {e}")
                    preserve_image_metadata_and_save(input_path, output_path, image)
            else:
                # 通常の保存（メタデータ保持）
                preserve_image_metadata_and_save(input_path, output_path, image)

    # CPU集約的な処理を別スレッドで実行
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, process_sync)


@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """処理済み画像を取得する"""
    file_path = PROCESSED_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(path=file_path, media_type="image/png", filename=filename)


@app.get("/api/uploads/{filename}")
async def get_uploaded_image(filename: str):
    """アップロード済み画像を取得する"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(path=file_path, media_type="image/png", filename=filename)


@app.post("/api/download-batch")
async def download_batch_images(filenames: List[str]):
    """複数の処理済み画像をZIPファイルでダウンロードする"""
    if not filenames:
        raise HTTPException(status_code=400, detail="No files specified")

    # ZIPファイルをメモリに作成
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename in filenames:
            file_path = PROCESSED_DIR / filename
            if file_path.exists():
                zip_file.write(file_path, filename)

    zip_buffer.seek(0)

    return StreamingResponse(
        io.BytesIO(zip_buffer.read()),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=processed_images.zip"},
    )


@app.delete("/api/clear-files")
async def clear_files():
    """アップロードファイルと処理済みファイルをクリアする"""
    try:
        upload_files = list(UPLOAD_DIR.glob("*"))
        processed_files = list(PROCESSED_DIR.glob("*"))

        # アップロードファイルを削除
        for file_path in upload_files:
            if file_path.is_file():
                file_path.unlink()

        # 処理済みファイルを削除
        for file_path in processed_files:
            if file_path.is_file():
                file_path.unlink()

        return {
            "message": "Files cleared successfully",
            "deleted_uploads": len(upload_files),
            "deleted_processed": len(processed_files),
            "total_deleted": len(upload_files) + len(processed_files),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear files: {str(e)}")


@app.delete("/api/clear-old-files")
async def clear_old_files(hours: int = 24):
    """指定時間以上古いファイルを削除する"""
    try:
        import time

        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)  # 指定時間前

        upload_files = list(UPLOAD_DIR.glob("*"))
        processed_files = list(PROCESSED_DIR.glob("*"))

        deleted_uploads = 0
        deleted_processed = 0

        # 古いアップロードファイルを削除
        for file_path in upload_files:
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_uploads += 1

        # 古い処理済みファイルを削除
        for file_path in processed_files:
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_processed += 1

        return {
            "message": f"Old files (older than {hours} hours) cleared successfully",
            "deleted_uploads": deleted_uploads,
            "deleted_processed": deleted_processed,
            "total_deleted": deleted_uploads + deleted_processed,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear old files: {str(e)}"
        )


@app.get("/api/storage-info")
async def get_storage_info():
    """ストレージ使用量情報を取得する"""
    try:
        import os

        def get_directory_size(path):
            total_size = 0
            file_count = 0
            if path.exists():
                for file_path in path.glob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_count += 1
            return total_size, file_count

        upload_size, upload_count = get_directory_size(UPLOAD_DIR)
        processed_size, processed_count = get_directory_size(PROCESSED_DIR)

        def format_size(size_bytes):
            if size_bytes == 0:
                return "0 B"
            size_names = ["B", "KB", "MB", "GB"]
            i = 0
            while size_bytes >= 1024 and i < len(size_names) - 1:
                size_bytes /= 1024
                i += 1
            return f"{size_bytes:.1f} {size_names[i]}"

        return {
            "uploads": {
                "size": upload_size,
                "size_formatted": format_size(upload_size),
                "file_count": upload_count,
            },
            "processed": {
                "size": processed_size,
                "size_formatted": format_size(processed_size),
                "file_count": processed_count,
            },
            "total": {
                "size": upload_size + processed_size,
                "size_formatted": format_size(upload_size + processed_size),
                "file_count": upload_count + processed_count,
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get storage info: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    from datetime import datetime

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "api_name": "Image Editor API",
    }


# SVG処理用のクラス
class SVGGenerateRequest(BaseModel):
    prefecture: str
    color: str = "#ff6277"


class SVGResponse(BaseModel):
    svg_content: str
    filename: str


# 都道府県リスト（SVGのid属性と対応）
PREFECTURES = [
    "北海道",
    "青森",
    "岩手",
    "宮城",
    "秋田",
    "山形",
    "福島",
    "茨城",
    "栃木",
    "群馬",
    "埼玉",
    "千葉",
    "東京",
    "神奈川",
    "新潟",
    "富山",
    "石川",
    "福井",
    "山梨",
    "長野",
    "岐阜",
    "静岡",
    "愛知",
    "三重",
    "滋賀",
    "京都",
    "大阪",
    "兵庫",
    "奈良",
    "和歌山",
    "鳥取",
    "島根",
    "岡山",
    "広島",
    "山口",
    "徳島",
    "香川",
    "愛媛",
    "高知",
    "福岡",
    "佐賀",
    "長崎",
    "熊本",
    "大分",
    "宮崎",
    "鹿児島",
    "沖縄",
]


@app.get("/api/prefectures")
async def get_prefectures():
    """都道府県リストを取得する"""
    return {"prefectures": PREFECTURES}


@app.post("/api/generate-svg", response_model=SVGResponse)
async def generate_svg(request: SVGGenerateRequest):
    """選択された都道府県をハイライトしたSVGを生成する"""
    try:
        svg_path = Path("assets/map.svg")

        if not svg_path.exists():
            raise HTTPException(status_code=404, detail="SVG template not found")

        # SVGファイルを読み込み
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()

        # まず、すべての都道府県の色を白色にリセット
        for prefecture in PREFECTURES:
            # 特別処理：SVG内のid属性と都道府県名のマッピング
            if prefecture == "徳島":
                reset_ids = ["徳島"]
            elif prefecture == "香川":
                reset_ids = ["徳島-2"]
            else:
                reset_ids = [prefecture]

            for prefecture_id in reset_ids:
                # 既存の色を白色にリセット
                pattern1 = f'id="{prefecture_id}"([^>]*?)fill=["\'](#[^"\']*)["\']'
                if re.search(pattern1, svg_content):
                    replacement1 = f'id="{prefecture_id}"\\1fill="#fff"'
                    svg_content = re.sub(pattern1, replacement1, svg_content)

        # 選択された都道府県の色を変更
        if request.prefecture in PREFECTURES:
            # 特別処理：SVG内のid属性と都道府県名のマッピング
            if request.prefecture == "徳島":
                prefecture_ids = ["徳島"]
            elif request.prefecture == "香川":
                # SVGファイル内では香川県がid="徳島-2"で表現されている
                prefecture_ids = ["徳島-2"]
            else:
                prefecture_ids = [request.prefecture]

            for prefecture_id in prefecture_ids:
                # より堅牢なSVG色変更処理
                # ケース1: 既存のfill属性がある場合（fill="#色" または fill='#色'）
                pattern1 = f'id="{prefecture_id}"([^>]*?)fill=["\'](#[^"\']*)["\']'
                if re.search(pattern1, svg_content):
                    replacement1 = f'id="{prefecture_id}"\\1fill="{request.color}"'
                    svg_content = re.sub(pattern1, replacement1, svg_content)
                    print(
                        f"Pattern 1 matched for {prefecture_id}: Updated existing fill attribute"
                    )
                    continue

                # ケース2: style属性内にfillがある場合
                pattern2 = (
                    f'id="{prefecture_id}"([^>]*?)style="([^"]*?)fill:[^;"]*([^"]*)"'
                )
                if re.search(pattern2, svg_content):
                    replacement2 = (
                        f'id="{prefecture_id}"\\1style="\\2fill:{request.color}\\3"'
                    )
                    svg_content = re.sub(pattern2, replacement2, svg_content)
                    print(
                        f"Pattern 2 matched for {prefecture_id}: Updated fill in style attribute"
                    )
                    continue

                # ケース3: fill属性がない場合は追加
                pattern3 = f'id="{prefecture_id}"([^>]*?)(?=[ >])'
                if re.search(pattern3, svg_content):
                    replacement3 = f'id="{prefecture_id}"\\1 fill="{request.color}"'
                    svg_content = re.sub(pattern3, replacement3, svg_content)
                    print(
                        f"Pattern 3 matched for {prefecture_id}: Added new fill attribute"
                    )
                else:
                    print(f"Warning: No pattern matched for {prefecture_id}")

        # ファイル名を生成
        filename = f"japan_map_{request.prefecture}_{uuid.uuid4().hex[:8]}.svg"
        output_path = PROCESSED_DIR / filename

        # SVGファイルを保存
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

        print(f"SVG generated successfully: {filename}")
        return SVGResponse(svg_content=svg_content, filename=filename)

    except Exception as e:
        print(f"SVG generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SVG generation failed: {str(e)}")


@app.get("/api/svg/{filename}")
async def get_svg(filename: str):
    """生成されたSVGファイルを取得する"""
    file_path = PROCESSED_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="SVG file not found")

    return FileResponse(path=file_path, media_type="image/svg+xml", filename=filename)


class FileInfo(BaseModel):
    filename: str
    size: int
    size_formatted: str
    created_at: str
    file_type: str


class FileListResponse(BaseModel):
    uploads: List[FileInfo]
    processed: List[FileInfo]
    total_count: int


@app.get("/api/files/list", response_model=FileListResponse)
async def get_file_list():
    """ファイル一覧を取得する"""
    try:
        import os
        from datetime import datetime

        def format_size(size_bytes):
            if size_bytes == 0:
                return "0 B"
            size_names = ["B", "KB", "MB", "GB"]
            i = 0
            while size_bytes >= 1024 and i < len(size_names) - 1:
                size_bytes /= 1024
                i += 1
            return f"{size_bytes:.1f} {size_names[i]}"

        def get_file_info(file_path, file_type):
            stat = file_path.stat()
            return FileInfo(
                filename=file_path.name,
                size=stat.st_size,
                size_formatted=format_size(stat.st_size),
                created_at=datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                file_type=file_type,
            )

        uploads = []
        processed = []

        # アップロード済みファイルを取得
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                uploads.append(get_file_info(file_path, "upload"))

        # 処理済みファイルを取得
        for file_path in PROCESSED_DIR.glob("*"):
            if file_path.is_file():
                processed.append(get_file_info(file_path, "processed"))

        # 作成日時順でソート（新しいものから）
        uploads.sort(key=lambda x: x.created_at, reverse=True)
        processed.sort(key=lambda x: x.created_at, reverse=True)

        return FileListResponse(
            uploads=uploads,
            processed=processed,
            total_count=len(uploads) + len(processed),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get file list: {str(e)}"
        )


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """個別ファイルを削除する"""
    try:
        upload_path = UPLOAD_DIR / filename
        processed_path = PROCESSED_DIR / filename

        deleted = False
        file_type = None

        if upload_path.exists():
            upload_path.unlink()
            deleted = True
            file_type = "upload"
        elif processed_path.exists():
            processed_path.unlink()
            deleted = True
            file_type = "processed"

        if not deleted:
            raise HTTPException(status_code=404, detail="File not found")

        return {
            "message": "File deleted successfully",
            "filename": filename,
            "file_type": file_type,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


# 文字・画像置換用のリクエストモデル
class TextReplaceRequest(BaseModel):
    text_replacements: Dict[str, str]
    image_replacements: Dict[str, str]  # image_id -> uploaded_image_url
    config: Optional[Dict] = None


@app.post("/api/replace-text-images")
async def replace_text_images(request: TextReplaceRequest):
    """テンプレート画像の文字と画像を置換する - 新しいテンプレートベースアプローチ"""
    try:
        # 置換用画像を読み込み
        image_replacements = {}
        for image_id, image_url in request.image_replacements.items():
            image_filename = image_url.split("/")[-1]
            image_path = UPLOAD_DIR / image_filename

            if image_path.exists():
                with Image.open(image_path) as replacement_image:
                    image_replacements[image_id] = replacement_image.copy()

        # 新しいテンプレートベース合成処理
        replacer = TextImageReplacer()
        result_image = replacer.create_composite_image(
            request.text_replacements,
            image_replacements
        )

        # 結果を保存（JPG形式）
        output_filename = f"text_replaced_{uuid.uuid4().hex[:8]}.jpg"
        output_path = PROCESSED_DIR / output_filename
        replacer.save_as_jpg(result_image, str(output_path))

        return {
            "message": "Text and images replaced successfully using template-based approach",
            "processed_url": f"/processed/{output_filename}",
            "filename": output_filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text replacement failed: {str(e)}")


@app.post("/api/create-tourism-layout")
async def create_tourism_layout(
    title: str,
    locations: List[str],
    image_urls: List[str]
):
    """観光パンフレット風のレイアウトを作成する"""
    try:
        # アップロードされた画像を読み込み
        images = []
        for image_url in image_urls:
            image_filename = image_url.split("/")[-1]
            image_path = UPLOAD_DIR / image_filename

            if image_path.exists():
                with Image.open(image_path) as img:
                    images.append(img.copy())

        if not images:
            raise HTTPException(status_code=400, detail="No valid images found")

        # パンフレットレイアウトを作成
        replacer = TextImageReplacer()
        layout_image = replacer.create_tourism_layout(title, locations, images)

        # 結果を保存（JPG形式）
        output_filename = f"tourism_layout_{uuid.uuid4().hex[:8]}.jpg"
        output_path = PROCESSED_DIR / output_filename
        replacer.save_as_jpg(layout_image, str(output_path))

        return {
            "message": "Tourism layout created successfully",
            "processed_url": f"/processed/{output_filename}",
            "filename": output_filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Layout creation failed: {str(e)}")


@app.get("/api/template-config")
async def get_template_config():
    """テンプレート設定を取得する"""
    return create_template_config()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

```
