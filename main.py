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
        result = self.draw_matches(img_left, img_right,keypoints_l, keypoints_r, matches)

        # Step 4 - fit the homography model with the RANSAC algorithm
        homography = self.find_homography(matches)
        """
        # Step 5 - Warp images to create the panoramic image
        result = self.warping(img_left, img_right, homography, ...)  # Add input arguments as you deem fit
        """
        return result

    def compute_descriptors(self, img):
        '''
        The feature detector and descriptor
        '''
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        orb = cv.ORB_create()
        keypoints = orb.detect(gray, None)
        keypoints, features = orb.compute(gray, keypoints)


        return keypoints, features

    def matching(self, descriptors_l, descriptors_r, cross_check=True):
        # Add input arguments as you deem fit for refinement
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

        #bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        #matches = bf.match(descriptors_l, descriptors_r)

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

        img_with_correspondesces = np.concatenate((image1, image2), axis=1)
        for match in matches:
            pt1 = keypoints1[match[0]].pt
            pt2 = keypoints2[match[1]].pt
            pt2 = (int(pt2[0]) + image1.shape[1], int(pt2[1]))
            cv.line(img_with_correspondesces, (int(pt1[0]), int(pt1[1])), pt2, (0, 255, 0), 1)
        return img_with_correspondesces


    def find_homography(self, matches):
        '''
        Fit the best homography model with the RANSAC algorithm.
        '''
        num_iterations = 100
        inlier_threshold = 3.0

        best_homography = None
        best_num_inliers = 0

        for _ in range(num_iterations):
            subset_indices = np.random.choice(len(matches), 4, replace=False)
            source_points = np.array([matches[i][0] for i in subset_indices])
            destination_points = np.array([matches[i][1] for i in subset_indices])


            homography = Homography().solve_homography(source_points, destination_points)

            num_inliers = 0
            for match in matches:
                source_point = np.array(matches[0][:2], dtype=np.float32)
                destination_point = np.array(matches[1][:2], dtype=np.float32)
                transformed_point = np.dot(homography, np.append(source_point, 1))
                if np.linalg.norm(transformed_point[:2] - destination_point) < inlier_threshold:
                    num_inliers += 1

            if num_inliers > best_num_inliers:
                best_homography = homography
                best_num_inliers = num_inliers
        return homography

    def warping(img_left, img_right, homography):  # Add input arguments as you deem fit
        '''
           Warp images to create panoramic image
        '''

        # Your code here. You will have to warp one image into another via the
        # homography. Remember that the homography is an entity expressed in
        # homogeneous coordinates.

        return result

    def remove_black_border(self, img):
        '''
        Remove black border after stitching
        '''
        return cropped_image


class Blender:
    def linear_blending(self):
        '''
        linear blending (also known as feathering)
        '''

        return linear_blending_img

    def customised_blending(self):
        '''
        Customised blending of your choice
        '''
        return customised_blending_img


class Homography:
    def solve_homography(self, S, D):
        '''
        Find the homography matrix between a set of S points and a set of
        D points
        '''

        if len(S) != 4 or len(D) != 4:
            raise ValueError("At elast 4 correspondences are needed to compute homography")

        S = np.reshape(S, (2, 2))
        D = np.reshape(D, (2, 2))

        A = []
        for i in range(2):
            x, y = S[i]
            u, v = D[i]
            A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
            A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])

        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        return H/H[2, 2]


if __name__ == "__main__":
    # Read the image files
    img_left = cv.imread('s1.jpg')
    img_right = cv.imread('s2.jpg')

    stitcher = Stitcher()
    result = stitcher.stitch(img_left, img_right)  # Add input arguments as you deem fit

    # show the result
    cv.imshow('result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
