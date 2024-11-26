import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2

def plot_fly_ellipse(row, cap, rotate=0, bodyparts=False, img=None, plot=True, figax=None, color="red"):
    """
    Plots the fly's ellipse and body parts on the frame extracted from the video.

    Parameters:
    - row: A pandas Series representing a row from df_pose with necessary columns.
    - cap: An OpenCV VideoCapture object for reading frames from the video.

    This function assumes that df_pose contains columns:
    - 'x', 'y', 'major', 'minor', 'angle', 'frame_number'
    - Body part coordinates: 'head_x', 'head_y', 'abdomen_x', 'abdomen_y', 'mLL_x', 'mLL_y', 'mRL_x', 'mRL_y'
    """
    # Set the frame position and read the frame
    if img is None:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_number"]))
        ret, img = cap.read()
        if not ret:
            print(f"Failed to read frame {row['frame_number']}")
            return

    # Check if the image is grayscale or color
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Image is grayscale; no need to convert
        img_rgb = img  # Ensure it's a 2D array

    height, width = img.shape[:2]

    if figax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        # Display the image with correct extent
        ax.imshow(img_rgb, extent=[0, width, height, 0])
    else:
        fig, ax = figax

    # Create the Ellipse object using the updated columns
    ellipse = Ellipse(
        (row['x'], row['y']),  # Use original y for plotting
        width=row['major'],      # Full length of major axis
        height=row['minor'],     # Full length of minor axis
        angle=row['angle']+rotate,      # Angle in degrees
        edgecolor=color, fc='None', lw=2
    )

    # Add the ellipse to the plot
    ax.add_patch(ellipse)

    if bodyparts:
        # Plot body parts
        ax.plot(row['head_x'], row['head_y'], 'bo', label='Head')  # Head
        ax.plot(row['abdomen_x'], row['abdomen_y'], 'go', label='Abdomen')  # Abdomen
        ax.plot([row['head_x'], row['abdomen_x']], [row['head_y'], row['abdomen_y']], 'r--', label='Major Axis')  # Major axis
        ax.plot(row['mLL_x'], row['mLL_y'], 'co', label='Middle Left Leg')  # Middle left leg
        ax.plot(row['mRL_x'], row['mRL_y'], 'mo', label='Middle Right Leg')  # Middle right leg
        ax.plot([row['mLL_x'], row['mRL_x']], [row['mLL_y'], row['mRL_y']], 'k--', label='Minor Axis')  # Minor axis

        # Adjust plot limits to focus on the fly
        # Note: Because the image origin is at the top-left, y-axis increases downward
        ax.set_xlim(row['x'] - row['major'] * 1.5, row['x'] + row['major'] * 1.5)
        ax.set_ylim(row['y'] + row['major'] * 1.5, row['y'] - row['major'] * 1.5)  # Swap limits to invert y-axis

        # Optional: Add legend and title
        ax.legend()
    ax.set_title(f"Fly ID: {row['id']} at Frame {int(row['frame_number'])}")
    if plot:
        plt.show()
    return (fig, ax), img
    
