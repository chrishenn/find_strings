import matplotlib.patches
import matplotlib.pyplot as plt


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
    plt.savefig('/home/chris/Documents/find_strings/data/hand_drawn_q_arcs/test/' + 'im_0' + '.png')



if __name__ == "__main__":
    gen_quarter()