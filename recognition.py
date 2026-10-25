
import cv2


def recognize_object(reference, query, min_matches=25):

    # 1. Convert images to grayscale
    reference_gray = cv2.cvtColor(
        reference, cv2.COLOR_BGR2GRAY
    )
    query_gray = cv2.cvtColor(
        query, cv2.COLOR_BGR2GRAY
    )

    # 2. Create SIFT detector
    sift = cv2.SIFT_create()

    # 3. Detect landmarks and extract descriptors
    kp_ref, desc_ref = sift.detectAndCompute(
        reference_gray, None
    )
    kp_query, desc_query = sift.detectAndCompute(
        query_gray, None
    )

    print("\n--- LANDMARK DETECTION ---")
    print("Reference landmarks:", len(kp_ref))
    print("Query landmarks:", len(kp_query))

    good_matches = []

    # 4. Match features using Brute-Force Matcher
    if desc_ref is not None and desc_query is not None:
        if len(desc_ref) > 0 and len(desc_query) >= 2:

            matcher = cv2.BFMatcher(cv2.NORM_L2)

            # KNN matching with k = 2
            matches = matcher.knnMatch(
                desc_ref, desc_query, k=2
            )

            # 5. Apply Lowe's ratio test
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair

                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

    print("\n--- FEATURE MATCHING ---")
    print("Good landmark matches:", len(good_matches))

    # 6. Determine recognition result
    if len(good_matches) >= min_matches:
        result = "OBJECT RECOGNIZED"
        color = (0, 255, 0)
    else:
        result = "OBJECT NOT RECOGNIZED"
        color = (0, 0, 255)

    print("Recognition Result:", result)

    # 7. Draw detected landmarks
    reference_landmarks = cv2.drawKeypoints(
        reference,
        kp_ref,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    query_landmarks = cv2.drawKeypoints(
        query,
        kp_query,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # 8. Draw matching features
    match_image = cv2.drawMatches(
        reference,
        kp_ref,
        query,
        kp_query,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Add result text to image
    cv2.putText(
        match_image,
        result,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
        cv2.LINE_AA
    )

    # 9. Save output images
    cv2.imwrite(
        "reference_landmarks.jpg",
        reference_landmarks
    )

    cv2.imwrite(
        "query_landmarks.jpg",
        query_landmarks
    )

    cv2.imwrite(
        "landmark_matches.jpg",
        match_image
    )

    return result, match_image
