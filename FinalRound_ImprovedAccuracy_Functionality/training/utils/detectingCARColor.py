from sklearn.cluster import KMeans
import cv2
import PIL
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import image as img1
import pandas as pd
from scipy.cluster.vq import whiten
import os

class DominantColors:
    CLUSTERS = None
    IMAGEPATH = None
    IMAGE = None
    COLORS = None
    LABELS = None
    BASEWIDTH = 256

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGEPATH = image


    def dominantColors(self):

        # read image
        img = cv2.imread(self.IMAGEPATH)

        # resize image
        imgh, imgw, _ = img.shape
        wpercent = (self.BASEWIDTH / float(imgw))
        hsize = int((float(imgh) * float(wpercent)))
        img = cv2.resize(img, (self.BASEWIDTH, hsize), PIL.Image.ANTIALIAS)

        # convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)

        # reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        # save image after operations
        self.IMAGE = img

        # using k-means to cluster pixels
        kmeans = KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(img)

        # the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_

        # save labels
        self.LABELS = kmeans.labels_

        # returning after converting to integer from float
        return self.COLORS.astype(int)


    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


    def analyseRGB(self):
        r = []
        g = []
        b = []
        image = img1.imread(self.IMAGEPATH)
        for line in image:
            for pixel in line:
                # print(pixel)
                temp_r, temp_g, temp_b = pixel
                r.append(temp_r)
                g.append(temp_g)
                b.append(temp_b)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(r, g, b)
        plt.show()
        df = pd.DataFrame({'red': r, 'blue': b, 'green': g})
        df['scaled_red'] = whiten(df['red'])
        df['scaled_blue'] = whiten(df['blue'])
        df['scaled_green'] = whiten(df['green'])
        df.sample(n=10)
        from scipy.cluster.vq import kmeans
        cluster_centers, distortion = kmeans(df[['scaled_red', 'scaled_green', 'scaled_blue']], 2)
        print(cluster_centers)
        colors = []
        r_std, g_std, b_std = df[['red', 'green', 'blue']].std()
        for cluster_center in cluster_centers:
            scaled_r, scaled_g, scaled_b = cluster_center
            colors.append((scaled_r * r_std / 255, scaled_g * g_std / 255, scaled_b * b_std / 255))
        plt.imshow([colors])
        plt.show()


    def plotClusters(self):
        # plotting
        fig = plt.figure()
        ax = Axes3D(fig)
        for label, pix in zip(self.LABELS, self.IMAGE):
            ax.scatter(pix[0], pix[1], pix[2], color=self.rgb_to_hex(self.COLORS[label]))
        plt.show()


    def plotHistogram(self):
        # labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS + 1)

        # create frequency count tables
        (hist, _) = np.histogram(self.LABELS, bins=numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()

        # appending frequencies to cluster centers
        colors = self.COLORS

        # descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()]

        # creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0

        # creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500

            # getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]

            # using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r, g, b), -1)
            start = end

            # display chart
        plt.figure()
        plt.axis("off")
        plt.imshow(chart)
        plt.show()

def _main_():
    clusters = 8
    for img in sorted(os.listdir('output\\predicted\\')):
        print(img)
        dc = DominantColors('..\\..\\data\\output\\predicted\\{0}'.format(img), clusters)
        colors = dc.dominantColors()
        dc.analyseRGB()

if __name__ == '__main__':
    _main_()


