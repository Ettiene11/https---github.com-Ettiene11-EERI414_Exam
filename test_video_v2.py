import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin
import csv

def get_fps_and_resolution(video_path):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Get the frame rate
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # Get the resolution
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Release the video capture object
    video_capture.release()
    
    return fps, (width, height)

def image_to_freq_signals(image_path):
    # Read the image in grayscale
    img = image_path
    
    # Take the Fourier Transform
    f_transform = np.fft.fft2(img)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    return img, f_transform_shifted

def create_circular_mask(shape, cutoff_radius):
    """Creates a circular mask with the given cutoff radius."""
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - center_y)**2 + (y - center_x)**2)
    mask = distance <= cutoff_radius
    return mask

def apply_circular_mask(f_transform_shifted, mask):
    return f_transform_shifted * mask

def reconstruct_image_from_freq_signals(filtered_transform_shifted):
    # Inverse shift the Fourier transform
    filtered_transform = np.fft.ifftshift(filtered_transform_shifted)
    
    # Inverse Fourier Transform to get the image back
    filtered_image = np.fft.ifft2(filtered_transform)
    filtered_image = np.abs(filtered_image)
    
    return filtered_image

def find_white_dot_center(image):
    # Convert the image to grayscale if it's a color image
    if len(image.shape) == 3:
        img_gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        img_gray = image.astype(np.uint8)
    
    # Get the dimensions of the image
    height, width = img_gray.shape
    
    # Iterate through each pixel to find the white dot
    for y in range(height):
        for x in range(width):
            if img_gray[y, x] == 255:  # White pixel (assuming 255 is white in grayscale)
                return (x, y)  # Return the coordinates of the white dot
    
    # If no white dot is found
    return None

def main():
    video_path = "Xfiles.mp4"  # Path to the video file
    fps, resolution = get_fps_and_resolution(video_path)
    print("Frames per second:", fps)
    print("Resolution:", resolution)

    video_capture = cv2.VideoCapture(video_path)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "UFO_tracking2.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    csv_file_path = "UFO_coordinates2.csv"
    csv_file = open(csv_file_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'x_coord', 'y_coord'])  # Write the header

    frame_number = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # Break the loop if there are no more frames
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Example usage
        # image_path = "path_to_your_image.jpg"
        # image_ndarray = np.array(Image.open(image_path))  # Load image as ndarray
        original_image, f_transform_shifted = image_to_freq_signals(gray)

        # Parameters for the FIR lowpass filter
        cutoff = 0.42  # Normalized cutoff frequency (0 < cutoff < 0.5)
        # cutoff = 0.5 
        # Determine the cutoff radius for the circular mask
        cutoff_radius = int(cutoff * min(original_image.shape) // 2)

        # Create and apply the circular mask
        circular_mask = create_circular_mask(f_transform_shifted.shape, cutoff_radius)
        filtered_transform_shifted = apply_circular_mask(f_transform_shifted, circular_mask)

        # Reconstruct the filtered image
        filtered_image = reconstruct_image_from_freq_signals(filtered_transform_shifted)

        blurred1 = filtered_image

        threshold = 40 #0.5
        Filtered1 = np.where(blurred1 < threshold, 255, 0)
        
        gray = Filtered1

        white_dot_center = find_white_dot_center(gray)
        if white_dot_center:
            print('Frame: ', frame_number)
            print("UFO location:", white_dot_center)
        else:
            print("No location detected.")

        if white_dot_center:
            csv_writer.writerow([frame_number, white_dot_center[0], white_dot_center[1]])
            cv2.circle(frame, white_dot_center, radius=5, color=(0, 0, 255), thickness=2)

        plt.figure(1)
        # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
        plt.imshow(gray)
        plt.axis('off')  # Turn off axis
        if white_dot_center:
            plt.plot(white_dot_center[0], white_dot_center[1], 'o', markerfacecolor='none', markeredgecolor='r', markersize=15)  # Plot a red dot
        plt.show(block=False)
        plt.pause(0.001)

        out.write(frame)

        frame_number += 1

        # Break the loop if 'Q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    csv_file.close()
    # Release the video capture object
    video_capture.release()
    out.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


