import cv2, os, argparse

def extract_frames(video_path, output_folder="frames"):
    if not os.path.isfile(video_path):
        print(f"Error: Video not found: {video_path}")
        return
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(fname, frame)
        count += 1

    cap.release()
    print(f"Extracted {count} frames to '{output_folder}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", default="frames", help="Output folder")
    args = parser.parse_args()
    extract_frames(args.video, args.output)
