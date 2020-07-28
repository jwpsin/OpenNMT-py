from errno import EEXIST
from os import makedirs, path


def autolabel(rects, ax, vert_off=(0, 3), font=8):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{0:.1f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=vert_off,  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom').set_fontsize(font)


def create_directory(mypath):
    """
    Creates a directory. equivalent to using mkdir -p on the command line
    """
    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise ValueError("Error in creating the directory")
