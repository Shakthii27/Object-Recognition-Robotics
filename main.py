
import cv2
import os

from calibration import undistort_image
from recognition import recognize_object


def main():
    # Image filenames
    reference_path = "reference.jpg"
    query_path = "query.jpg"

    # Check whether input images exist
    if not os.path.exists(reference_path):
        print("Error: reference.jpg not found!")
        return

    if not os.path.exists(query_path):
        print("Error: query.jpg not found!")
        return

    # Load images
    reference = cv2.imread(reference_path)
    query = cv2.imread(query_path)

    if reference is None or query is None:
        print("Error: Unable to load input images.")
        return

    print("Images loaded successfully.")

    # Step 1: Camera calibration and undistortion
    undistorted_query, camera_matrix, dist_coeffs = (
        undistort_image(query)
    )

    cv2.imwrite(
        "undistorted_query.jpg",
        undistorted_query
    )

    # Step 2: Detect landmarks and recognize object
    result, match_image = recognize_object(
        reference,
        undistorted_query
    )

    # Step 3: Display images
    cv2.imshow("Reference Image", reference)
    cv2.imshow("Original Query Image", query)
    cv2.imshow("Undistorted Query Image", undistorted_query)

    cv2.imshow(
        "Landmark Matching and Recognition",
        match_image
    )

    print("\nOutput images saved successfully.")
    print(" - undistorted_query.jpg")
    print(" - reference_landmarks.jpg")
    print(" - query_landmarks.jpg")
    print(" - landmark_matches.jpg")

    # Wait until a key is pressed
    print("\nPress any key on an image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
