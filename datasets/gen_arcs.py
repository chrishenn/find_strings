import matplotlib.patches
import matplotlib.pyplot as plt

from PIL import Image


def gen_quarter():
    dpi = 1
    figure, ax = plt.subplots(dpi=dpi, figsize=[600, 600])
    # arc_artist = matplotlib.patches.Arc((300,300), 200, 200, theta1=0, theta2=90, lw=200)
    arc_artist = matplotlib.patches.Arc((300,300), 200, 200, theta1=90, theta2=180, lw=200)

    ax.set_aspect('equal')
    ax.set_ylim(0, 600)
    ax.set_xlim(0, 600)
    ax.axis('off')

    ax.add_artist(arc_artist)

    plt.show(block=False)
    # plt.savefig('/home/chris/Documents/find_strings/data/hand_drawn_q_arcs/test/' + 'im_5' + '.png')


def gen_circle():
    dpi = 1
    figure, ax = plt.subplots(dpi=dpi, figsize=[600, 600])

    ax.set_aspect('equal')
    ax.set_ylim(0, 600)
    ax.set_xlim(0, 600)
    ax.axis('off')

    circle = plt.Circle((400., 400.), 100)
    ax.add_artist(circle)

    plt.show(block=False)
    plt.savefig('/home/chris/Documents/find_strings/data/art_arcs/test/' + 'im_6' + '.png')


def rot_existing():

    img = Image.open('/home/chris/Documents/find_strings/data/art_arcs/test/blob_0.png')
    img = img.rotate(145, expand=False, resample=2, fillcolor=255)

    img.save('/home/chris/Documents/find_strings/data/art_arcs/test/' + 'blob_3' + '.png')

if __name__ == "__main__":
    # gen_quarter()
    # gen_circle()
    rot_existing()