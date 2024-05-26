import numpy as np
import matplotlib.pyplot as plt

image_filename = "./frames/frame_0004.png"

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def calculate_2dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

def calculate_distance_from_centre(coords, centre):
    # Distance from centre is √(x^2 + y^2)
    return np.sqrt(
        (coords[0] - centre) ** 2 + (coords[1] - centre) ** 2
    )

def find_symmetric_coordinates(coords, centre):
    return (centre + (centre - coords[0])), (centre + (centre - coords[1]))

def display_plots(individual_grating, reconstruction, idx, freq_sin):
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(individual_grating)
    plt.title("Grating")
    plt.axis("off")
    plt.subplot(222)
    plt.imshow(reconstruction)
    plt.title("Reconstruction")
    plt.axis("off")
    plt.subplot(223)
    plt.plot(freq_sin)
    plt.title("1D Sinusoid")
    plt.xlabel("Pixel")
    plt.ylabel("Intensity")
    plt.subplot(224)
    plt.imshow(np.log(abs(ft)))
    plt.title("FFT")
    plt.axis("off")
    plt.suptitle(f"Terms: {idx}")
    plt.pause(1)

# Read and process image
image = plt.imread(image_filename)
image = image[:, :, :3].mean(axis=2)  # Convert to grayscale

# Array dimensions (array is square) and centre pixel
array_size = len(image)
centre = int((array_size - 1) / 2)

# Get all coordinate pairs in the left half of the array,
# including the column at the centre of the array (which
# includes the centre pixel)
coords_left_half = (
    (x, y) for x in range(array_size) for y in range(centre+1)
)

# Sort points based on distance from centre
coords_left_half = sorted(
    coords_left_half,
    key=lambda x: calculate_distance_from_centre(x, centre),
    # reverse=True
)

plt.set_cmap("gray")

ft = calculate_2dft(image)

# Show grayscale image and its Fourier transform
plt.subplot(121)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
plt.subplot(122)
plt.imshow(np.log(abs(ft)))
plt.title("FFT")
plt.axis("off")
plt.pause(2)

# Reconstruct image
fig = plt.figure()

# Set up empty arrays for final image and
# individual gratings
rec_image = np.zeros(image.shape)
individual_grating = np.zeros(
    image.shape, dtype="complex"
)
idx = 0

# Step 2
nth_result = 1000  # Change this value to the desired iteration number

for coords in coords_left_half:
    # Central column: only include if points in top half of
    # the central column
    if not (coords[1] == centre and coords[0] > centre):
        idx += 1
        symm_coords = find_symmetric_coordinates(
            coords, centre
        )
        # Step 3
        # Copy values from Fourier transform into
        # individual_grating for the pair of points in
        # current iteration
        individual_grating[coords] = ft[coords]
        individual_grating[symm_coords] = ft[symm_coords]

        # Step 4
        # Calculate inverse Fourier transform to give the
        # reconstructed grating. Add this reconstructed
        # grating to the reconstructed image
        rec_grating = calculate_2dift(individual_grating)
        rec_image += rec_grating

        # Clear individual_grating array, ready for
        # next iteration
        individual_grating[coords] = 0
        individual_grating[symm_coords] = 0

        # Calculate 1D sinusoid for the current frequency grating
        freq_sin = np.real(np.fft.fftshift(coords))
        display_plots(rec_grating, rec_image, idx, freq_sin)

plt.show()

