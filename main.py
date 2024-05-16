import numpy as np
import cv2 as cv


class Stitcher:
    def __init__(self):
        pass

    def stitch(self, img_left, img_right):  # Add input arguments as you deem fit
        '''
            The main method for stitching two images
        '''

        keypoints_l, descriptors_l = self.compute_descriptors(img_left)
        keypoints_r, descriptors_r = self.compute_descriptors(img_right)

        matches = self.matching(descriptors_l, descriptors_r)  # Add input arguments as you deem fit

        print("Number of matching correspondences selected:", len(matches))

        # Step 3 - Draw the matches connected by lines
        result = self.draw_matches(img_left, img_right, keypoints_l, keypoints_r, matches)

        # Step 4 - fit the homography model with the RANSAC algorithm
        homography = self.find_homography(matches, keypoints_l, keypoints_r)
        """
        # Step 5 - Warp images to create the panoramic image
        result = self.warping(img_left, img_right, homography, ...)  # Add input arguments as you deem fit
        """
        warped_image = self.warping(img_left, img_right, homography)
        removed_border = self.remove_black_border(warped_image)
        blended_image = Blender().linear_blending(removed_border, img_left, overlap_width=10)
        return result, blended_image

    def compute_descriptors(self, img):
        '''
        The feature detector and descriptor
        '''
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        orb = cv.ORB_create()
        keypoints = orb.detect(gray, None)
        keypoints, features = orb.compute(gray, keypoints)

        keypoints = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        return keypoints, features

    def matching(self, descriptors_l, descriptors_r, cross_check=True):
        '''
        Find the matching correspondences between the two images
        '''
        matches = []
        for i, desc_l in enumerate(descriptors_l):
            best_dist = float('inf')
            best_idx = -1
            for j, desc_r in enumerate(descriptors_r):
                dist = np.linalg.norm(desc_l - desc_r)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j
            if cross_check:
                if i == self.get_best_match_index(descriptors_r[best_idx], descriptors_l):
                    matches.append((i, best_idx))
            else:
                matches.append((i, best_idx))

        return matches

    def get_best_match_index(self, desc, descriptors):
        best_dist = float('inf')
        best_index = -1
        for i, desc_ref in enumerate(descriptors):
            dist = np.linalg.norm(desc - desc_ref)
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return best_index

    def draw_matches(self, image1, image2, keypoints1, keypoints2, matches):
        '''
            Connect correspondences between images with lines and draw these
            lines
        '''

        img_with_correspondences = np.concatenate((image1, image2), axis=1)
        for match in matches:
            pt1 = keypoints1[match[0]]
            pt2 = keypoints2[match[1]]
            pt2 = (int(pt2[0]) + image1.shape[1], int(pt2[1]))
            cv.line(img_with_correspondences, (int(pt1[0]), int(pt1[1])), pt2, (0, 255, 0), 1)

            cv.circle(img_with_correspondences, (int(pt1[0]), int(pt1[1])), radius=3, color=(255, 0, 0), thickness=-1)
            cv.circle(img_with_correspondences, pt2, radius=3, color=(255, 0, 0), thickness=-1)
        return img_with_correspondences

    def find_homography(self, matches, keypoints1, keypoints2):
        '''
        Fit the best homography model with the RANSAC algorithm.
        '''
        num_iterations = 1000
        inlier_threshold = 3.0

        best_homography = np.identity(3)
        best_num_inliers = 0

        for _ in range(num_iterations):
            num_inliers = 0
            subset_indices = np.random.choice(len(matches) - 1, 4, replace=False)
            source_points = np.array([(matches[i][0], matches[i][1]) for i in subset_indices])
            destination_points = np.array([(matches[i + 1][0], matches[i + 1][1]) for i in subset_indices])

            homography = Homography().solve_homography(source_points, destination_points)

            for source_index, destination_index in matches:
                if source_index in subset_indices and destination_index in subset_indices:
                    source_index_in_subset = np.where(subset_indices == source_index)[0][0]
                    destination_index_in_subset = np.where(subset_indices == destination_index)[0][0]
                    source_point = source_points[source_index_in_subset]
                    destination_point = destination_points[destination_index_in_subset]
                    transformed_point = np.dot(homography, np.append(source_point, 1))
                    transformed_point = transformed_point / transformed_point[-1]

                    distance = np.sqrt(np.sum((transformed_point[:2] - destination_point) ** 2))
                    if 0 < distance < inlier_threshold:
                        num_inliers += 1

                    if num_inliers > best_num_inliers:
                        best_num_inliers = num_inliers
                        best_homography = homography

        return best_homography

    def warping(self, img_left, img_right, homography):
        '''
           Warp images to create panoramic image
        '''
        height, width = img_left.shape[:2]
        warped_image = np.zeros_like(img_left, dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                src_x, src_y, src_w = np.dot(homography, [x, y, 1])
                src_x /= src_w
                src_y /= src_w

                if 0 <= src_x < img_right.shape[1] and 0 <= src_y < img_right.shape[0]:
                    warped_image[y, x] = img_right[int(src_y), int(src_x)]

        canvas_width = img_left.shape[1] + warped_image.shape[1]
        canvas_height = max(img_left.shape[0], warped_image.shape[0])
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        canvas[:img_left.shape[0], :img_left.shape[1]] = img_left

        canvas[:warped_image.shape[0], img_left.shape[1]:] = warped_image

        # This part is a test for homography.
        warped_img = cv.warpPerspective(img_right, homography, (width, height))
        overlay_img = cv.addWeighted(img_left, 0.5, warped_img, 0.5, 0)
        cv.imshow("overlay", overlay_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return canvas

    def remove_black_border(self, img):
        '''
        Remove black border after stitching
        '''
        # Convert the image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find contours of non-black regions
        contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Find the bounding box of the largest contour
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(largest_contour)

            # Crop the image using the bounding box
            cropped_image = img[y:y + h, x:x + w]

            return cropped_image
        else:
            # If no contours found, return the original image
            return img


class Blender:
    def linear_blending(self, img_left, img_right, overlap_width):
        '''
        Linear blending (feathering) of two images
        '''

        mask = np.zeros_like(img_left)
        mask[:, :overlap_width] = 1.0

        img_right_resized = cv.resize(img_right, (img_left.shape[1], img_left.shape[0]))

        blended_overlap = cv.addWeighted(img_left, 0.5, img_right_resized, 0.5, 0)

        linear_blending_img = np.where(mask == 1, blended_overlap, img_left)

        return linear_blending_img

    def customised_blending(self):
        '''
        Customised blending of your choice
        '''
        return customised_blending_img


class Homography:
    def solve_homography(self, S, D):
        '''
        Find the homography matrix between a set of source points and a set of
        destination points using the Direct Linear Transform (DLT) method
        '''

        if len(S) != 4 or len(D) != 4:
            raise ValueError("At least 4 correspondences are needed to compute homography Source_points has len",
                             len(S), " Destination_points has len", len(D))

        S = np.reshape(S, (4, 2))
        D = np.reshape(D, (4, 2))

        A = []
        for i in range(4):
            x, y = S[i]
            u, v = D[i]
            A.append([-u, -v, -1, 0, 0, 0, x * u, y * u, u])
            A.append([0, 0, 0, -u, -v, -1, x * v, y * v, v])

        A = np.array(A)
        U, s, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H = H / H[2, 2]

        return H


if __name__ == "__main__":
    # Read the image files
    img_left = cv.imread('s1.jpg')
    img_right = cv.imread('s2.jpg')

    stitcher = Stitcher()
    result, stitched = stitcher.stitch(img_left, img_right)  # Add input arguments as you deem fit

    # show the result
    cv.imshow('result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('Warped Image', stitched)
    cv.waitKey(0)
    cv.destroyAllWindows()
