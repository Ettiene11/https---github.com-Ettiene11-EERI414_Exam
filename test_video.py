import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv

import torch
torch.cuda.is_available()



torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

def transform_and_reconstruct(Nstart, Nstop, X, mu, sorted_eigvec_descending, window_size, num_windows_h, num_windows_w, step):
    Nwindow = window_size**2
    A = np.zeros((Nwindow, Nwindow))
    for n in range(Nstart, Nstop):
        A[n, :] = sorted_eigvec_descending[:, n]

    y = A @ X
    xk = A.T @ y
    for n in range(X.shape[1]):
        xk[:, n] = xk[:, n] + mu

    out = np.zeros((num_windows_h * step + window_size, num_windows_w * step + window_size))
    k = 0
    for i in range(num_windows_h):
        for j in range(num_windows_w):
            out[i * step:i * step + window_size, j * step:j * step + window_size] = xk[:, k].reshape(window_size, window_size)
            k += 1

    return out

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

    output_path = "UFO_tracking.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    csv_file_path = "UFO_coordinates.csv"
    csv_file = open(csv_file_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'x_coord', 'y_coord'])  # Write the header

    frame_number = 0

    while True:
        import numpy as np
        ret, frame = video_capture.read()
        if not ret:
            break  # Break the loop if there are no more frames
        
        beeld = frame

        gray = 0.3*beeld[:,:,0] + 0.59*beeld[:,:,1] + 0.11*beeld[:,:,2]

        kernel = 1/32 * np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]])

        blurred1 = cv2.filter2D(beeld, -1, kernel)

        threshold = 15 #0.08
        Filtered1 = np.where(blurred1 < threshold, 255, blurred1)

        gray = 0.3*Filtered1[:,:,0] + 0.59*Filtered1[:,:,1] + 0.11*Filtered1[:,:,2]

        # window_size = 10
        # step = 10 

        # num_windows_h = (gray.shape[0] - window_size) // step + 1
        # num_windows_w = (gray.shape[1] - window_size) // step + 1

        # X = np.zeros((window_size * window_size, num_windows_h * num_windows_w))

        # idx = 0
        # for i in range(num_windows_h):
        #     for j in range(num_windows_w):
        #         window = gray[i * step:i * step + window_size, j * step:j * step + window_size].reshape(1, -1)
        #         X[:, idx] = window
        #         idx += 1

        # (Nx, Ny) = X.shape
        # mu = np.mean(X, 1)
        # for n in range(Ny):
        #     X[:, n] = X[:, n] - mu

        # Cx = np.cov(X, rowvar='true')
        # eigval, eigvec = np.linalg.eig(Cx)
        # descending_indices = np.argsort(eigval)[::-1]
        # sorted_eigvec_descending = eigvec[:, descending_indices]
        # KumSom = np.cumsum(eigval)
        # KumSom = 100 * KumSom / np.amax(KumSom)

        # startindex = 1
        # stopindex = 1
        # image_result = transform_and_reconstruct(startindex-1, stopindex, X, mu, sorted_eigvec_descending, window_size, num_windows_h, num_windows_w, step)

        # Display the result
        # plt.imshow(Filtered1, cmap='gray')
        # plt.title('Eigenvectors used: ' + str(startindex) + '->' + str(stopindex))
        # plt.draw()
        # plt.pause(0.001)

        import numpy as np



        # Example usage
        # image_path = "path_to_your_image.jpg"
        # image_ndarray = np.array(Image.open(image_path))  # Load image as ndarray
        white_dot_center = find_white_dot_center(gray)
        if white_dot_center:
            print('Frame: ', frame_number)
            print("UFO location:", white_dot_center)
        else:
            print("No location detected.")

        if white_dot_center:
            csv_writer.writerow([frame_number, white_dot_center[0], white_dot_center[1]])
            cv2.circle(frame, white_dot_center, radius=5, color=(0, 0, 255), thickness=2)

        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
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

