import gradio as gr
import os
import cv2
import numpy as np
import shutil
import subprocess
import time
from SinglePhoto import FaceSwapper
import argparse

wellcomingMessage = """
    <h1>Face Swapping Suite</h1>
    <p>All-in-one face swapping: single photo, video, multi-source, and multi-destination!</p>
"""

swapper = FaceSwapper()

#Photo Swapping Functions
def swap_single_photo(src_img, src_idx, dst_img, dst_idx, progress=gr.Progress(track_tqdm=True)):
    log = ""
    start_time = time.time()
    try:
        progress(0, desc="Preparing files")
        src_path = "SinglePhoto/data_src.jpg"
        dst_path = "SinglePhoto/data_dst.jpg"
        output_path = "SinglePhoto/output_swapped.jpg"
        os.makedirs(os.path.dirname(src_path), exist_ok=True)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
        dst_img_bgr = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(src_path, src_img_bgr)
        cv2.imwrite(dst_path, dst_img_bgr)
        log += f"Saved source to {src_path}, destination to {dst_path}\n"
        progress(0.5, desc="Swapping faces")
        result = swapper.swap_faces(src_path, int(src_idx), dst_path, int(dst_idx))
        cv2.imwrite(output_path, result)
        log += f"Swapped and saved result to {output_path}\n"
        progress(0.8, desc="Cleaning up")
        try:
            if os.path.exists(src_path):
                os.remove(src_path)
            if os.path.exists(dst_path):
                os.remove(dst_path)
            log += "Cleaned up temp files.\n"
        except Exception as cleanup_error:
            log += f"Cleanup error: {cleanup_error}\n"
        progress(1, desc="Done")
        elapsed = time.time() - start_time
        log += f"Elapsed time: {elapsed:.2f} seconds\n"
        return output_path, log
    except Exception as e:
        log += f"Error: {e}\n"
        progress(1, desc="Error")
        elapsed = time.time() - start_time
        log += f"Elapsed time: {elapsed:.2f} seconds\n"
        return None, log

def swap_single_src_multi_dst(src_img, dst_imgs, dst_indices, progress=gr.Progress(track_tqdm=True)):
    log = ""
    results = []
    src_dir = "SingleSrcMultiDst/src"
    dst_dir = "SingleSrcMultiDst/dst"
    output_dir = "SingleSrcMultiDst/output"
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(dst_img, tuple):
        dst_img = dst_img[0]
    dst_img_bgr = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
    dst_path = os.path.join(dst_dir, "data_dst.jpg")
    cv2.imwrite(dst_path, dst_img_bgr)
    log += f"Saved destination image to {dst_path}\n"
    progress(0.05, desc="Saved destination image")

    for i, src_img in enumerate(src_imgs):
        if isinstance(src_img, tuple):
            src_img = src_img[0]
        src_path = os.path.join(src_dir, f"data_src_{i}.jpg")
        output_path = os.path.join(output_dir, f"output_swapped_{i}.jpg")
        src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(src_path, src_img_bgr)
        log += f"Saved source image {i} to {src_path}\n"
        try:
            result = swapper.swap_faces(src_path, 1, dst_path, int(dst_idx))
            cv2.imwrite(output_path, result)
            results.append(output_path)
            log += f"Swapped and saved result to {output_path}\n"
        except Exception as e:
            results.append(f"Error: {e}")
            log += f"Error swapping source {i}: {e}\n"
        progress((i + 1) / len(src_imgs), desc=f"Swapping source {i+1}/{len(src_imgs)}")
    progress(1, desc="Done")
    return results, log

def swap_multi_src_single_dst(src_imgs, dst_img, dst_idx, progress=gr.Progress(track_tqdm=True)):
    log = ""
    results = []
    src_dir = "MultiSrcSingleDst/src"
    dst_dir = "MultiSrcSingleDst/dst"
    output_dir = "MultiSrcSingleDst/output"
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(dst_img, tuple):
        dst_img = dst_img[0]
    dst_img_bgr = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
    dst_path = os.path.join(dst_dir, "data_dst.jpg")
    cv2.imwrite(dst_path, dst_img_bgr)
    log += f"Saved destination image to {dst_path}\n"
    progress(0.05, desc="Saved destination image")

    for i, src_img in enumerate(src_imgs):
        if isinstance(src_img, tuple):
            src_img = src_img[0]
        src_path = os.path.join(src_dir, f"data_src_{i}.jpg")
        output_path = os.path.join(output_dir, f"output_swapped_{i}.jpg")
        src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(src_path, src_img_bgr)
        log += f"Saved source image {i} to {src_path}\n"
        try:
            result = swapper.swap_faces(src_path, 1, dst_path, int(dst_idx))
            cv2.imwrite(output_path, result)
            results.append(output_path)
            log += f"Swapped and saved result to {output_path}\n"
        except Exception as e:
            results.append(f"Error: {e}")
            log += f"Error swapping source {i}: {e}\n"
        progress((i + 1) / len(src_imgs), desc=f"Swapping source {i+1}/{len(src_imgs)}")
    progress(1, desc="Done")
    return results, log

def swap_multi_src_multi_dst(src_imgs, dst_imgs, dst_indices, progress=gr.Progress(track_tqdm=True)):
    log = ""
    results = []
    src_dir = "MultiSrcMultiDst/src"
    dst_dir = "MultiSrcMultiDst/dst"
    output_dir = "MultiSrcMultiDst/output"
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(dst_indices, str):
        dst_indices_list = [int(idx.strip()) for idx in dst_indices.split(",") if idx.strip().isdigit()]
    else:
        dst_indices_list = [int(idx) for idx in dst_indices]

    total = max(1, len(src_imgs) * len(dst_imgs))
    count = 0
    for i, src_img in enumerate(src_imgs):
        if isinstance(src_img, tuple):
            src_img = src_img[0]
        if src_img is None:
            results.append(f"Error: Source image at index {i} is None")
            log += f"Source image at index {i} is None\n"
            continue
        src_path = os.path.join(src_dir, f"data_src_{i}.jpg")
        if isinstance(src_img, np.ndarray):
            src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(src_path, src_img_bgr)
            log += f"Saved source image {i} to {src_path}\n"
        elif isinstance(src_img, str) and os.path.exists(src_img):
            shutil.copy(src_img, src_path)
            log += f"Copied source image {i} from {src_img} to {src_path}\n"
        else:
            results.append(f"Error: Invalid source image at index {i}")
            log += f"Invalid source image at index {i}\n"
            continue
        for j, dst_img in enumerate(dst_imgs):
            if isinstance(dst_img, tuple):
                dst_img = dst_img[0]
            if dst_img is None:
                results.append(f"Error: Destination image at index {j} is None")
                log += f"Destination image at index {j} is None\n"
                continue
            dst_path = os.path.join(dst_dir, f"data_dst_{j}.jpg")
            output_path = os.path.join(output_dir, f"output_swapped_{i}_{j}.jpg")
            if isinstance(dst_img, np.ndarray):
                dst_img_bgr = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(dst_path, dst_img_bgr)
                log += f"Saved destination image {j} to {dst_path}\n"
            elif isinstance(dst_img, str) and os.path.exists(dst_img):
                shutil.copy(dst_img, dst_path)
                log += f"Copied destination image {j} from {dst_img} to {dst_path}\n"
            else:
                results.append(f"Error: Invalid destination image at index {j}")
                log += f"Invalid destination image at index {j}\n"
                continue
            try:
                dst_idx = dst_indices_list[j] if j < len(dst_indices_list) else 1
                result = swapper.swap_faces(src_path, 1, dst_path, int(dst_idx))
                cv2.imwrite(output_path, result)
                results.append(output_path)
                log += f"Swapped src {i} with dst {j} and saved to {output_path}\n"
            except Exception as e:
                results.append(f"Error: {e}")
                log += f"Error swapping src {i} with dst {j}: {e}\n"
            count += 1
            progress(count / total, desc=f"Swapping ({count}/{total})")
    progress(1, desc="Done")
    return results, log

def swap_single_src_multi_dst(src_img, dst_imgs, dst_indices, progress=gr.Progress(track_tqdm=True)):
    log = ""
    results = []
    src_dir = "SingleSrcMultiDst/src"
    dst_dir = "SingleSrcMultiDst/dst"
    output_dir = "SingleSrcMultiDst/output"
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(src_img, tuple):
        src_img = src_img[0]
    src_path = os.path.join(src_dir, "data_src.jpg")
    src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(src_path, src_img_bgr)
    log += f"Saved source image to {src_path}\n"
    progress(0.05, desc="Saved source image")

    if isinstance(dst_indices, str):
        dst_indices_list = [int(idx.strip()) for idx in dst_indices.split(",") if idx.strip().isdigit()]
    else:
        dst_indices_list = [int(idx) for idx in dst_indices]

    for j, dst_img in enumerate(dst_imgs):
        if isinstance(dst_img, tuple):
            dst_img = dst_img[0]
        dst_path = os.path.join(dst_dir, f"data_dst_{j}.jpg")
        output_path = os.path.join(output_dir, f"output_swapped_{j}.jpg")
        dst_img_bgr = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst_path, dst_img_bgr)
        log += f"Saved destination image {j} to {dst_path}\n"
        try:
            dst_idx = dst_indices_list[j] if j < len(dst_indices_list) else 1
            result = swapper.swap_faces(src_path, 1, dst_path, int(dst_idx))
            cv2.imwrite(output_path, result)
            results.append(output_path)
            log += f"Swapped and saved result to {output_path}\n"
        except Exception as e:
            results.append(f"Error: {e}")
            log += f"Error swapping with destination {j}: {e}\n"
        progress((j + 1) / len(dst_imgs), desc=f"Swapping destination {j+1}/{len(dst_imgs)}")
    progress(1, desc="Done")
    return results, log

def swap_faces_custom(src_imgs, dst_img, mapping_str, progress=gr.Progress(track_tqdm=True)):
    """
    src_imgs: list of source images (numpy arrays)
    dst_img: destination image (numpy array)
    mapping_str: comma-separated string, e.g. "2,1,3"
    """
    log = ""
    start_time = time.time()
    dst_path = "CustomSwap/data_dst.jpg"
    output_path = "CustomSwap/output_swapped.jpg"
    src_dir = "CustomSwap/src"
    temp_dir = "CustomSwap/temp"
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save destination image
    dst_img_bgr = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dst_path, dst_img_bgr)
    log += f"Saved destination image to {dst_path}\n"

    # Save all source images
    src_paths = []
    for i, src_img in enumerate(src_imgs):
        src_path = os.path.join(src_dir, f"data_src_{i+1}.jpg")
        if isinstance(src_img, tuple):
            src_img = src_img[0]
        if src_img is None:
            log += f"Source image {i+1} is None, skipping.\n"
            continue
        if isinstance(src_img, np.ndarray):
            src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(src_path, src_img_bgr)
            src_paths.append(src_path)
            log += f"Saved source image {i+1} to {src_path}\n"
        elif isinstance(src_img, str) and os.path.exists(src_img):
            shutil.copy(src_img, src_path)
            src_paths.append(src_path)
            log += f"Copied source image {i+1} from {src_img} to {src_path}\n"
        else:
            log += f"Source image {i+1} is not a valid image, skipping.\n"

    # Parse mapping
    try:
        mapping = [int(x.strip()) for x in mapping_str.split(",") if x.strip().isdigit()]
    except Exception as e:
        log += f"Error parsing mapping: {e}\n"
        elapsed = time.time() - start_time
        log += f"Elapsed time: {elapsed:.2f} seconds\n"
        return None, log

    # Use a temp file for iterative swapping
    temp_dst_path = os.path.join(temp_dir, "temp_dst.jpg")
    shutil.copy(dst_path, temp_dst_path)

    for face_idx, src_idx in enumerate(mapping, start=1):
        if src_idx < 1 or src_idx > len(src_paths):
            log += f"Invalid source index {src_idx} for face {face_idx}, skipping.\n"
            continue
        try:
            swapped_img = swapper.swap_faces(src_paths[src_idx-1], 1, temp_dst_path, face_idx)
            cv2.imwrite(temp_dst_path, swapped_img)
            log += f"Swapped face {face_idx} in destination with source {src_idx}\n"
        except Exception as e:
            log += f"Failed to swap face {face_idx} with source {src_idx}: {e}\n"

    shutil.copy(temp_dst_path, output_path)
    log += f"Saved swapped image to {output_path}\n"
    if os.path.exists(temp_dst_path):
        os.remove(temp_dst_path)
    elapsed = time.time() - start_time
    log += f"Elapsed time: {elapsed:.2f} seconds\n"
    return output_path, log

#Video Swapping Functions
def add_audio_to_video(original_video_path, video_no_audio_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_no_audio_path,
        "-i", original_video_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode()

def swap_video(src_img, src_idx, video, dst_idx, delete_frames_dir=True, add_audio=True, copy_to_drive=False, progress=gr.Progress()):
    log = ""
    start_time = time.time()
    src_path = "VideoSwapping/data_src.jpg"
    dst_video_path = "VideoSwapping/data_dst.mp4"
    frames_dir = "VideoSwapping/video_frames"
    swapped_dir = "VideoSwapping/swapped_frames"
    output_video_path = "VideoSwapping/output_tmp_output_video.mp4"
    final_output_path = "VideoSwapping/output_with_audio.mp4"

    os.makedirs(os.path.dirname(src_path), exist_ok=True)
    os.makedirs(os.path.dirname(dst_video_path), exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(swapped_dir, exist_ok=True)

    src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(src_path, src_img_bgr)
    log += f"Saved source image to {src_path}\n"
    progress(0.05, desc="Saved source image")

    if isinstance(video, str) and os.path.exists(video):
        shutil.copy(video, dst_video_path)
        log += f"Copied video to {dst_video_path}\n"
    else:
        dst_video_path = video

    from VideoSwapping import extract_frames, frames_to_video

    frame_paths = extract_frames(dst_video_path, frames_dir)
    log += f"Extracted {len(frame_paths)} frames to {frames_dir}\n"
    progress(0.15, desc="Extracted frames")

    swapped_files = set(os.listdir(swapped_dir))
    total_frames = len(frame_paths)
    start_loop_time = time.time()
    for idx, frame_path in enumerate(frame_paths):
        swapped_name = f"swapped_{idx:05d}.jpg"
        out_path = os.path.join(swapped_dir, swapped_name)
        if swapped_name in swapped_files and os.path.exists(out_path):
            log += f"Frame {idx}: already swapped, skipping.\n"
            elapsed = time.time() - start_loop_time
            avg_time = elapsed / (idx + 1) if idx + 1 > 0 else 0
            remaining = avg_time * (total_frames - (idx + 1))
            mins, secs = divmod(int(remaining), 60)
            progress(0.15 + 0.6 * (idx + 1) / total_frames, desc=f"Swapping {idx+1}/{total_frames} | {mins:02d}:{secs:02d} left")
            continue
        try:
            try:
                swapped = swapper.swap_faces(src_path, int(src_idx), frame_path, int(dst_idx))
            except ValueError as ve:
                if int(dst_idx) != 1 and "Target image contains" in str(ve):
                    swapped = swapper.swap_faces(src_path, int(src_idx), frame_path, 1)
                    log += f"Frame {idx}: dst_idx {dst_idx} not found, used 1 instead.\n"
                else:
                    raise ve
            cv2.imwrite(out_path, swapped)
            log += f"Swapped frame {idx} and saved to {out_path}\n"
        except Exception as e:
            cv2.imwrite(out_path, cv2.imread(frame_path))
            log += f"Failed to swap frame {idx}: {e}\n"
        elapsed = time.time() - start_loop_time
        avg_time = elapsed / (idx + 1) if idx + 1 > 0 else 0
        remaining = avg_time * (total_frames - (idx + 1))
        mins, secs = divmod(int(remaining), 60)
        progress(0.15 + 0.6 * (idx + 1) / total_frames, desc=f"Swapping {idx+1}/{total_frames} | {mins:02d}:{secs:02d} left")
    cap = cv2.VideoCapture(dst_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    frames_to_video(swapped_dir, output_video_path, fps)
    log += f"Combined swapped frames into video {output_video_path}\n"
    progress(0.8, desc="Muxing audio")

    # Copy to Google Drive if requested
    if copy_to_drive:
        drive_path = "/content/drive/MyDrive/" + os.path.basename(output_video_path)
        try:
            shutil.copy(output_video_path, drive_path)
            log += f"Copied swapped video without audio to Google Drive: {drive_path}\n"
        except Exception as e:
            log += f"Failed to copy to Google Drive: {e}\n"

    if add_audio:
        ok, audio_log = add_audio_to_video(dst_video_path, output_video_path, final_output_path)
        if ok:
            log += f"Added audio to {final_output_path}\n"
        else:
            log += f"Audio muxing failed: {audio_log}\n"
            final_output_path = output_video_path
    else:
        final_output_path = output_video_path
        log += "Audio was not added as per user request.\n"

    try:
        if os.path.exists(src_path):
            os.remove(src_path)
        if os.path.exists(dst_video_path):
            os.remove(dst_video_path)
        if delete_frames_dir and os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
            log += "Deleted video_frames directory.\n"
        elif not delete_frames_dir:
            log += "Kept video_frames directory as requested.\n"
        if os.path.exists(swapped_dir):
            shutil.rmtree(swapped_dir)
        log += "Cleaned up temp files and folders.\n"
    except Exception as cleanup_error:
        log += f"Cleanup error: {cleanup_error}\n"
    progress(1, desc="Done")
    elapsed = time.time() - start_time
    log += f"Elapsed time: {elapsed:.2f} seconds\n"
    return final_output_path, log

def swap_video_all_faces(src_img, video, num_faces_to_swap, delete_frames_dir=True, add_audio=True, copy_to_drive=False, progress=gr.Progress()):
    log = ""
    start_time = time.time()
    src_path = "VideoSwappingAllFaces/data_src.jpg"
    dst_video_path = "VideoSwappingAllFaces/data_dst.mp4"
    frames_dir = "VideoSwappingAllFaces/video_frames"
    swapped_dir = "VideoSwappingAllFaces/swapped_frames"
    output_video_path = "VideoSwappingAllFaces/output_tmp_output_video.mp4"
    final_output_path = "VideoSwappingAllFaces/output_with_audio.mp4"

    os.makedirs(os.path.dirname(src_path), exist_ok=True)
    os.makedirs(os.path.dirname(dst_video_path), exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(swapped_dir, exist_ok=True)

    src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(src_path, src_img_bgr)
    log += f"Saved source image to {src_path}\n"
    progress(0.05, desc="Saved source image")

    if isinstance(video, str) and os.path.exists(video):
        shutil.copy(video, dst_video_path)
        log += f"Copied video to {dst_video_path}\n"
    else:
        dst_video_path = video

    from VideoSwapping import extract_frames, frames_to_video

    frame_paths = extract_frames(dst_video_path, frames_dir)
    log += f"Extracted {len(frame_paths)} frames to {frames_dir}\n"
    progress(0.15, desc="Extracted frames")

    swapped_files = set(os.listdir(swapped_dir))
    temp_dir = os.path.join(swapped_dir, "temp_swap")
    os.makedirs(temp_dir, exist_ok=True)
    total_frames = len(frame_paths)
    start_loop_time = time.time()

    for idx, frame_path in enumerate(frame_paths):
        swapped_name = f"swapped_{idx:05d}.jpg"
        out_path = os.path.join(swapped_dir, swapped_name)
        temp_frame_path = os.path.join(temp_dir, "temp.jpg")
        if swapped_name in swapped_files and os.path.exists(out_path):
            log += f"Frame {idx}: already swapped, skipping.\n"
            elapsed = time.time() - start_loop_time
            avg_time = elapsed / (idx + 1) if idx + 1 > 0 else 0
            remaining = avg_time * (total_frames - (idx + 1))
            mins, secs = divmod(int(remaining), 60)
            progress(0.15 + 0.6 * (idx + 1) / total_frames, desc=f"Swapping {idx+1}/{total_frames} | {mins:02d}:{secs:02d} left")
            continue
        try:
            shutil.copy(frame_path, temp_frame_path)
            for face_idx in range(1, int(num_faces_to_swap) + 1):
                try:
                    swapped_img = swapper.swap_faces(src_path, 1, temp_frame_path, face_idx)
                    cv2.imwrite(temp_frame_path, swapped_img)
                except Exception as e:
                    log += f"Failed to swap face {face_idx} in frame {idx}: {e}\n"
            shutil.copy(temp_frame_path, out_path)
            log += f"Swapped all faces in frame {idx} and saved to {out_path}\n"
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
        except Exception as e:
            cv2.imwrite(out_path, cv2.imread(frame_path))
            log += f"Failed to swap frame {idx}: {e}\n"
        elapsed = time.time() - start_loop_time
        avg_time = elapsed / (idx + 1) if idx + 1 > 0 else 0
        remaining = avg_time * (total_frames - (idx + 1))
        mins, secs = divmod(int(remaining), 60)
        progress(0.15 + 0.6 * (idx + 1) / total_frames, desc=f"Swapping {idx+1}/{total_frames} | {mins:02d}:{secs:02d} left")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    cap = cv2.VideoCapture(dst_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    frames_to_video(swapped_dir, output_video_path, fps)
    log += f"Combined swapped frames into video {output_video_path}\n"
    progress(0.8, desc="Muxing audio")

    # Copy to Google Drive if requested
    if copy_to_drive:
        drive_path = "/content/drive/MyDrive/" + os.path.basename(output_video_path)
        try:
            shutil.copy(output_video_path, drive_path)
            log += f"Copied swapped video without audio to Google Drive: {drive_path}\n"
        except Exception as e:
            log += f"Failed to copy to Google Drive: {e}\n"

    if add_audio:
        ok, audio_log = add_audio_to_video(dst_video_path, output_video_path, final_output_path)
        if ok:
            log += f"Added audio to {final_output_path}\n"
        else:
            log += f"Audio muxing failed: {audio_log}\n"
            final_output_path = output_video_path
    else:
        final_output_path = output_video_path
        log += "Audio was not added as per user request.\n"

    try:
        if os.path.exists(src_path):
            os.remove(src_path)
        if os.path.exists(dst_video_path):
            os.remove(dst_video_path)
        if delete_frames_dir and os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
            log += "Deleted video_frames directory.\n"
        elif not delete_frames_dir:
            log += "Kept video_frames directory as requested.\n"
        if os.path.exists(swapped_dir):
            shutil.rmtree(swapped_dir)
        log += "Cleaned up temp files and folders.\n"
    except Exception as cleanup_error:
        log += f"Cleanup error: {cleanup_error}\n"
    progress(1, desc="Done")
    elapsed = time.time() - start_time
    log += f"Elapsed time: {elapsed:.2f} seconds\n"
    return final_output_path, log

def swap_video_custom_mapping(src_imgs, video, mapping_str, delete_frames_dir=True, add_audio=True, copy_to_drive=False, progress=gr.Progress()):
    log = ""
    start_time = time.time()
    src_dir = "CustomVideoSwap/src"
    temp_dir = "CustomVideoSwap/temp"
    frames_dir = "CustomVideoSwap/frames"
    swapped_dir = "CustomVideoSwap/swapped_frames"
    output_video_path = "CustomVideoSwap/output_tmp_output_video.mp4"
    final_output_path = "CustomVideoSwap/output_with_audio.mp4"
    dst_video_path = "CustomVideoSwap/data_dst.mp4"

    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(swapped_dir, exist_ok=True)

    # Save all source images
    src_paths = []
    for i, src_img in enumerate(src_imgs):
        src_path = os.path.join(src_dir, f"data_src_{i+1}.jpg")
        if isinstance(src_img, tuple):
            src_img = src_img[0]
        if src_img is None:
            log += f"Source image {i+1} is None, skipping.\n"
            continue
        if isinstance(src_img, np.ndarray):
            src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(src_path, src_img_bgr)
            src_paths.append(src_path)
            log += f"Saved source image {i+1} to {src_path}\n"
        elif isinstance(src_img, str) and os.path.exists(src_img):
            shutil.copy(src_img, src_path)
            src_paths.append(src_path)
            log += f"Copied source image {i+1} from {src_img} to {src_path}\n"
        else:
            log += f"Source image {i+1} is not a valid image, skipping.\n"

    # Parse mapping
    try:
        mapping = [int(x.strip()) for x in mapping_str.split(",") if x.strip().isdigit()]
    except Exception as e:
        log += f"Error parsing mapping: {e}\n"
        elapsed = time.time() - start_time
        log += f"Elapsed time: {elapsed:.2f} seconds\n"
        return None, log

    # Prepare video
    if isinstance(video, str) and os.path.exists(video):
        shutil.copy(video, dst_video_path)
        log += f"Copied video to {dst_video_path}\n"
    else:
        dst_video_path = video

    from VideoSwapping import extract_frames, frames_to_video

    frame_paths = extract_frames(dst_video_path, frames_dir)
    log += f"Extracted {len(frame_paths)} frames to {frames_dir}\n"
    progress(0.1, desc="Extracted frames")

    swapped_files = set(os.listdir(swapped_dir))
    temp_frame_path = os.path.join(temp_dir, "temp.jpg")
    total_frames = len(frame_paths)
    start_loop_time = time.time()

    for idx, frame_path in enumerate(frame_paths):
        swapped_name = f"swapped_{idx:05d}.jpg"
        out_path = os.path.join(swapped_dir, swapped_name)
        if swapped_name in swapped_files and os.path.exists(out_path):
            log += f"Frame {idx}: already swapped, skipping.\n"
            elapsed = time.time() - start_loop_time
            avg_time = elapsed / (idx + 1) if idx + 1 > 0 else 0
            remaining = avg_time * (total_frames - (idx + 1))
            mins, secs = divmod(int(remaining), 60)
            progress(0.1 + 0.7 * (idx + 1) / total_frames, desc=f"Swapping {idx+1}/{total_frames} | {mins:02d}:{secs:02d} left")
            continue
        try:
            shutil.copy(frame_path, temp_frame_path)
            for face_idx, src_idx in enumerate(mapping, start=1):
                if src_idx < 1 or src_idx > len(src_paths):
                    log += f"Invalid source index {src_idx} for face {face_idx} in frame {idx}, skipping.\n"
                    continue
                try:
                    swapped_img = swapper.swap_faces(src_paths[src_idx-1], 1, temp_frame_path, face_idx)
                    cv2.imwrite(temp_frame_path, swapped_img)
                    log += f"Frame {idx}: Swapped face {face_idx} with source {src_idx}\n"
                except Exception as e:
                    log += f"Frame {idx}: Failed to swap face {face_idx} with source {src_idx}: {e}\n"
            shutil.copy(temp_frame_path, out_path)
            log += f"Swapped all faces in frame {idx} and saved to {out_path}\n"
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
        except Exception as e:
            cv2.imwrite(out_path, cv2.imread(frame_path))
            log += f"Failed to swap frame {idx}: {e}\n"
        elapsed = time.time() - start_loop_time
        avg_time = elapsed / (idx + 1) if idx + 1 > 0 else 0
        remaining = avg_time * (total_frames - (idx + 1))
        mins, secs = divmod(int(remaining), 60)
        progress(0.1 + 0.7 * (idx + 1) / total_frames, desc=f"Swapping {idx+1}/{total_frames} | {mins:02d}:{secs:02d} left")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    cap = cv2.VideoCapture(dst_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    frames_to_video(swapped_dir, output_video_path, fps)
    log += f"Combined swapped frames into video {output_video_path}\n"
    progress(0.9, desc="Muxing audio")

    if add_audio:
        ok, audio_log = add_audio_to_video(dst_video_path, output_video_path, final_output_path)
        if ok:
            log += f"Added audio to {final_output_path}\n"
        else:
            log += f"Audio muxing failed: {audio_log}\n"
            final_output_path = output_video_path
    else:
        final_output_path = output_video_path
        log += "Audio was not added as per user request.\n"

    try:
        if os.path.exists(dst_video_path):
            os.remove(dst_video_path)
        if delete_frames_dir and os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
            log += "Deleted video_frames directory.\n"
        elif not delete_frames_dir:
            log += "Kept video_frames directory as requested.\n"
        if os.path.exists(swapped_dir):
            shutil.rmtree(swapped_dir)
        log += "Cleaned up temp files and folders.\n"
    except Exception as cleanup_error:
        log += f"Cleanup error: {cleanup_error}\n"
    progress(1, desc="Done")
    elapsed = time.time() - start_time
    log += f"Elapsed time: {elapsed:.2f} seconds\n"
    return final_output_path, log

def swap_single_src_multi_video(src_img, dst_videos, dst_indices, delete_frames_dir=True, add_audio=True, copy_to_drive=False, progress=gr.Progress(track_tqdm=True)):
    """
    Swaps a single source image onto multiple videos, with a mapping for destination face indices.
    Each video is processed one by one.
    """
    log = ""
    results = []
    start_time = time.time()
    base_dir = "SingleSrcMultiVideo"
    dst_dir = os.path.join(base_dir, "dst")
    frames_dir = os.path.join(base_dir, "video_frames")
    swapped_dir = os.path.join(base_dir, "swap_frames")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(swapped_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Parse indices
    if isinstance(dst_indices, str):
        dst_indices_list = [int(idx.strip()) for idx in dst_indices.split(",") if idx.strip().isdigit()]
    else:
        dst_indices_list = [int(idx) for idx in dst_indices]

    # Save source image
    src_path = os.path.join(base_dir, "data_src.jpg")
    src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(src_path, src_img_bgr)
    log += f"Saved source image to {src_path}\n"

    from VideoSwapping import extract_frames, frames_to_video

    for i, video in enumerate(dst_videos):
        dst_idx = dst_indices_list[i] if i < len(dst_indices_list) else 1
        video_name = f"video_{i}.mp4"
        dst_video_path = os.path.join(dst_dir, video_name)
        output_video_path = os.path.join(output_dir, f"output_{i}.mp4")
        output_video_with_audio = os.path.join(output_dir, f"output_with_audio_{i}.mp4")

        # Copy video to dst_dir
        if isinstance(video, str) and os.path.exists(video):
            shutil.copy(video, dst_video_path)
            log += f"Copied video {video} to {dst_video_path}\n"
        else:
            dst_video_path = video

        # Extract frames
        frame_paths = extract_frames(dst_video_path, frames_dir)
        log += f"Extracted {len(frame_paths)} frames from {dst_video_path} to {frames_dir}\n"
        progress(i / len(dst_videos), desc=f"Processing video {i+1}/{len(dst_videos)}")

        swapped_files = set(os.listdir(swapped_dir))
        total_frames = len(frame_paths)
        temp_frame_path = os.path.join(swapped_dir, "temp.jpg")
        start_loop_time = time.time()

        for idx, frame_path in enumerate(frame_paths):
            swapped_name = f"swapped_{idx:05d}.jpg"
            out_path = os.path.join(swapped_dir, swapped_name)
            if swapped_name in swapped_files and os.path.exists(out_path):
                continue
            try:
                shutil.copy(frame_path, temp_frame_path)
                try:
                    swapped_img = swapper.swap_faces(src_path, 1, temp_frame_path, int(dst_idx))
                    cv2.imwrite(temp_frame_path, swapped_img)
                except Exception as e:
                    log += f"Failed to swap face in frame {idx} of video {i}: {e}\n"
                shutil.copy(temp_frame_path, out_path)
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
            except Exception as e:
                cv2.imwrite(out_path, cv2.imread(frame_path))
                log += f"Failed to swap frame {idx} of video {i}: {e}\n"

        # Combine frames to video
        cap = cv2.VideoCapture(dst_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        frames_to_video(swapped_dir, output_video_path, fps)
        log += f"Combined swapped frames into video {output_video_path}\n"

        # Copy to Google Drive if requested
        if copy_to_drive:
            drive_path = "/content/drive/MyDrive/" + os.path.basename(output_video_path)
            try:
                shutil.copy(output_video_path, drive_path)
                log += f"Copied swapped video without audio to Google Drive: {drive_path}\n"
            except Exception as e:
                log += f"Failed to copy to Google Drive: {e}\n"

        # Mux audio if requested
        if add_audio:
            ok, audio_log = add_audio_to_video(dst_video_path, output_video_path, output_video_with_audio)
            if ok:
                log += f"Added audio to {output_video_with_audio}\n"
                results.append(output_video_with_audio)
            else:
                log += f"Audio muxing failed for video {i}: {audio_log}\n"
                results.append(output_video_path)
        else:
            results.append(output_video_path)
            log += f"Audio was not added for video {i} as per user request.\n"

        # Cleanup frames for next video
        if delete_frames_dir and os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
            os.makedirs(frames_dir, exist_ok=True)
            log += f"Deleted video_frames directory after video {i}.\n"
        elif not delete_frames_dir:
            log += f"Kept video_frames directory after video {i} as requested.\n"
        if os.path.exists(swapped_dir):
            shutil.rmtree(swapped_dir)
            os.makedirs(swapped_dir, exist_ok=True)
            log += f"Cleared swap_frames directory after video {i}.\n"

    elapsed = time.time() - start_time
    log += f"Elapsed time: {elapsed:.2f} seconds\n"
    return results, log



# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown(wellcomingMessage)
    with gr.Tab("Single Photo Swapping"):
        gr.Interface(
            fn=swap_single_photo,
            inputs=[
                gr.Image(label="Source Image"),
                gr.Number(value=1, label="Source Face Index"),
                gr.Image(label="Destination Image"),
                gr.Number(value=1, label="Destination Face Index"),
            ],
            outputs=[
                gr.Image(label="Swapped Image"),
                gr.Textbox(label="Log Output", lines=8, interactive=False)
            ],
        )
    
    with gr.Tab("SingleSrc MultiDst"):
        gr.Interface(
            fn=swap_single_src_multi_dst,
            inputs=[
                gr.Image(label="Source Image"),
                gr.Gallery(label="Destination Images", type="numpy", columns=3),
                gr.Textbox(label="Destination Face Indices (comma-separated, e.g. 1,1,2)"),
            ],
            outputs=[
                gr.Gallery(label="Swapped Images"),
                gr.Textbox(label="Log Output", lines=8, interactive=False)
            ],
        )
    with gr.Tab("MultiSrc SingleDst"):
        gr.Interface(
            fn=swap_multi_src_single_dst,
            inputs=[
                gr.Gallery(label="Source Images", type="numpy", columns=3),
                gr.Image(label="Destination Image"),
                gr.Number(value=1, label="Destination Face Index"),
            ],
            outputs=[
                gr.Gallery(label="Swapped Images"),
                gr.Textbox(label="Log Output", lines=8, interactive=False)
            ],
        )
    with gr.Tab("MultiSrc MultiDst"):
        gr.Interface(
            fn=swap_multi_src_multi_dst,
            inputs=[
                gr.Gallery(label="Source Images", type="numpy", columns=3),
                gr.Gallery(label="Destination Images", type="numpy", columns=3),
                gr.Textbox(label="Destination Face Indices (comma-separated, e.g. 1,1,2)"),
            ],
            outputs=[
                gr.Gallery(label="Swapped Images"),
                gr.Textbox(label="Log Output", lines=8, interactive=False)
            ],
        )
    with gr.Tab("Custom Face Mapping"):
        gr.Interface(
            fn=swap_faces_custom,
            inputs=[
                gr.Gallery(label="Source Images", type="numpy", columns=3),
                gr.Image(label="Destination Image"),
                gr.Textbox(label="Mapping (comma-separated, e.g. 2,1,3)"),
            ],
            outputs=[
                gr.Image(label="Swapped Image"),
                gr.Textbox(label="Log Output", lines=8, interactive=False)
            ],
        )
    with gr.Tab("Video Swapping"):
        gr.Interface(
            fn=swap_video,
            inputs=[
                gr.Image(label="Source Image"),
                gr.Number(value=1, label="Source Face Index"),
                gr.Video(label="Target Video"),
                gr.Number(value=1, label="Destination Face Index"),
                gr.Checkbox(label="Delete video_frames directory after processing", value=True),
                gr.Checkbox(label="Add audio from original video", value=True),
                gr.Checkbox(label="Copy swapped video (no audio) to Google Drive root", value=False)
            ],
            outputs=[
                gr.Video(label="Swapped Video"),
                gr.Textbox(label="Log Output", lines=8, interactive=False)
            ],
        )
    with gr.Tab("Video All Faces"):
        gr.Interface(
            fn=swap_video_all_faces,
            inputs=[
                gr.Image(label="Source Image"),
                gr.Video(label="Target Video"),
                gr.Number(value=1, label="Number of Faces to Swap"),
                gr.Checkbox(label="Delete video_frames directory after processing", value=True),
                gr.Checkbox(label="Add audio from original video", value=True),
                gr.Checkbox(label="Copy swapped video (no audio) to Google Drive root", value=False)
            ],
            outputs=[
                gr.Video(label="Swapped Video"),
                gr.Textbox(label="Log Output", lines=8, interactive=False)
            ],
        )
    with gr.Tab("Custom Video Face Mapping"):
        gr.Interface(
            fn=swap_video_custom_mapping,
            inputs=[
                gr.Gallery(label="Source Images", type="numpy", columns=3),
                gr.Video(label="Target Video"),
                gr.Textbox(label="Mapping (comma-separated, e.g. 2,1,3)"),
                gr.Checkbox(label="Delete video_frames directory after processing", value=True),
                gr.Checkbox(label="Add audio from original video", value=True),
                gr.Checkbox(label="Copy swapped video (no audio) to Google Drive root", value=False)
            ],
            outputs=[
                gr.Video(label="Swapped Video"),
                gr.Textbox(label="Log Output", lines=8, interactive=False)
            ],
        )
    with gr.Tab("SingleSrcMultiVideo"):
        gr.Interface(
            fn=swap_single_src_multi_video,
            inputs=[
                gr.Image(label="Source Image"),
                gr.File(label="Target Videos (select multiple)", file_count="multiple", type="filepath"),
                gr.Textbox(label="Destination Face Indices (comma-separated, e.g. 1,2,1)"),
                gr.Checkbox(label="Delete video_frames directory after processing", value=True),
                gr.Checkbox(label="Add audio from original video", value=True),
                gr.Checkbox(label="Copy swapped video (no audio) to Google Drive root", value=False)
            ],
            outputs=[
                gr.Gallery(label="Swapped Videos", type="filepath"),
                gr.Textbox(label="Log Output", lines=8, interactive=False)
            ],
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Launch Gradio with share=True")
    args = parser.parse_args()
    demo.launch(share=args.share)
