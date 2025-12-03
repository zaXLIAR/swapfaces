import cv2
import os
import shutil
import time
from SinglePhoto import FaceSwapper

def extract_frames(video_path, frames_dir):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
        last_idx = -1
    else:
        # Find the last extracted frame index
        existing = [f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".jpg")]
        if existing:
            last_idx = max([int(f.split("_")[1].split(".")[0]) for f in existing])
        else:
            last_idx = -1

    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f"frame_{idx:05d}.jpg")
        if idx > last_idx:
            cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        idx += 1
    cap.release()
    return frame_paths

def frames_to_video(frames_dir, output_video_path, fps):
    frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    if not frames:
        print("No frames found in directory.")
        return
    first_frame = cv2.imread(frames[0])
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()

def main():
    # Use files from VideoSwapping folder
    video_path = os.path.join("VideoSwapping", "data_dst.mp4")
    source_image_path = os.path.join("VideoSwapping", "data_src.jpg")
    frames_dir = os.path.join("VideoSwapping", "video_frames")
    swapped_dir = os.path.join("VideoSwapping", "swapped_frames")
    output_video_path = os.path.join("VideoSwapping", "output_swapped_video.mp4")
    source_face_idx = 1

    # Ask user for target_face_idx
    while True:
        try:
            user_input = input("Enter target_face_idx (default is 1): ").strip()
            if user_input == "":
                dest_face_idx = 1
                break
            dest_face_idx = int(user_input)
            break
        except ValueError:
            print("Invalid input. Please enter an integer value.")

    print("Choose an option:")
    print("1. Extract frames only")
    print("2. Face swap only (requires extracted frames)")
    print("3. Both extract frames and face swap")
    choice = input("Enter 1, 2, or 3: ").strip()

    frame_paths = []
    if choice == "1" or choice == "3":
        print("Extracting frames from video (resuming if needed)...")
        frame_paths = extract_frames(video_path, frames_dir)
        print(f"Extracted {len(frame_paths)} frames to {frames_dir}.")
        if choice == "1":
            return

    if choice == "2":
        # If only face swap, ensure frames are present
        if not os.path.exists(frames_dir):
            print("Frames directory does not exist. Please extract frames first.")
            return
        frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    if choice == "2" or choice == "3":
        # Prepare output directory
        if not os.path.exists(swapped_dir):
            os.makedirs(swapped_dir)

        # Initialize face swapper
        swapper = FaceSwapper()

        # Swap faces on each frame
        print("Swapping faces on frames...")
        start_time = time.time()
        for idx, frame_path in enumerate(frame_paths):
            frame_name = os.path.basename(frame_path)
            # Replace 'frame_' with 'swapped_' in the filename
            if frame_name.startswith("frame_"):
                swapped_name = "swapped_" + frame_name[len("frame_"):]
            else:
                swapped_name = "swapped_" + frame_name
            out_path = os.path.join(swapped_dir, swapped_name)
            if os.path.exists(out_path):
                # Skip already swapped frames
                print(f"Frame {idx+1}/{len(frame_paths)} already swapped, skipping...", end='\r')
                continue
            try:
                try:
                    swapped = swapper.swap_faces(
                        source_path=source_image_path,
                        source_face_idx=source_face_idx,
                        target_path=frame_path,
                        target_face_idx=dest_face_idx
                    )
                except ValueError as ve:
                    if "Target image contains" in str(ve):
                        print(f"\nFrame {idx}: Target face idx {dest_face_idx} not found, trying with idx 1.",end='\r')
                        swapped = swapper.swap_faces(
                            source_path=source_image_path,
                            source_face_idx=source_face_idx,
                            target_path=frame_path,
                            target_face_idx=1
                        )
                    else:
                        raise ve
                cv2.imwrite(out_path, swapped)
            except Exception as e:
                print(f"\nFrame {idx}: {e}")
                # Optionally, copy the original frame if swap fails
                cv2.imwrite(out_path, cv2.imread(frame_path))
            # Estimate time left
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(frame_paths) - (idx + 1))
            mins, secs = divmod(int(remaining), 60)
            print(f"Swapping frame {idx+1}/{len(frame_paths)} | Est. time left: {mins:02d}:{secs:02d}", end='\r')
        print()  # Move to the next line after the loop

        # Get FPS from original video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Combine swapped frames into video
        print("Combining swapped frames into video...")
        frames_to_video(swapped_dir, output_video_path, fps)
        print(f"Done! Output video saved as {output_video_path}")

        # Ask user if they want to keep the extracted frames and swapped images
        answer = input("Do you want to keep the extracted frames and swapped images? (y/n): ").strip().lower()
        if answer == 'n':
            try:
                shutil.rmtree(frames_dir)
                shutil.rmtree(swapped_dir)
                print("Temporary folders deleted.")
            except Exception as e:
                print(f"Error deleting folders: {e}")
        else:
            print("Temporary folders kept.")

if __name__ == "__main__":
    main()