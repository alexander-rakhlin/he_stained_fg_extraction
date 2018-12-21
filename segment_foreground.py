def segment_foreground(img):
    """Foreground Extraction for Histopathological Whole-Slide Imaging"""

    def isClose(_seeds, _start):
        MAX_DIST = 500
        for seed in _seeds:
            if np.sqrt(np.sum(np.square(np.array(_start) - np.array(seed[0])))) <= MAX_DIST:
                return True
        return False

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    chnl = [0, 1]
    lab[..., chnl] = 255
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.threshold(gray, np.mean(gray), maxval=255, type=cv2.THRESH_BINARY)[1]

    midsizeKernel = (5, 5)
    largeSigma = 1
    blurred = cv2.GaussianBlur(thresholded, midsizeKernel, largeSigma)
    mask = cv2.threshold(blurred, blurred.mean(), maxval=150, type=cv2.THRESH_BINARY)[1]
    inverse_mask = cv2.threshold(blurred, blurred.mean(), maxval=255, type=cv2.THRESH_BINARY_INV)[1]
    dseed = cv2.distanceTransform(inverse_mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)

    mask = cv2.medianBlur(mask.astype(np.uint8), ksize=7) + 100
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    max_point = np.unravel_index(np.argmax(dseed, axis=None), dseed.shape)[::-1]
    cv2.floodFill(mask, None, seedPoint=max_point, newVal=0)
    mask[mask > 0] = 255

    distance = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=0)
    final_mask = mask.copy()
    dmax = distance.max()
    globalMax = distance.max()
    seeds = []
    while dmax > 0:
        start = np.unravel_index(distance.argmax(), distance.shape)[::-1]
        if (dmax > 0.6 * globalMax) or isClose(seeds, start):
            cv2.floodFill(final_mask, None, start, newVal=200)
            seeds.append((start, dmax))
        cv2.floodFill(mask, None, seedPoint=start, newVal=0)
        distance[mask == 0] = 0
        dmax = distance.max()

    final_mask[final_mask != 200] = 0
    final_mask[final_mask == 200] = 255
    return final_mask
