import cv2
import os

def images_to_video(image_folder, output_video_path, frame_rate=10):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  # Ensure the images are in the correct order
    
    if not images:
        print("No images found in the specified folder.")
        return

    # Get the width and height of the first image
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the video writer object
    video_writer.release()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    # Set the path to the folder containing images and the output video path
    image_folder = '/SSDe/heeseon/src/C2Former/output/visualize/C2Former/0.5'
    output_video_path = '/SSDe/heeseon/src/C2Former/output/visualize/C2Former/video_0.5.mp4'
    
    # Convert images to video
    images_to_video(image_folder, output_video_path, frame_rate=10)

