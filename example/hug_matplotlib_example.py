"""Example web application for displaying a matplotlib image

This application can be run by entering this on the command line
    hug -f hug_matplotlib_example.py
and going to http://127.0.0.1:8000/?n=100 in a web browser.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import hug

temp = """
<img src="data:image/png;base64,{result}"\>
"""


def compute(n):
    a = np.random.rand(n)

    plt.plot(a)

    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    #figdata_png = base64.b64encode(figfile.read())
    figdata_png = base64.b64encode(figfile.getvalue())
    plt.close(plt.gcf())
    return figdata_png.decode('utf8')


@hug.get("/", output=hug.output_format.html)
def main(n: int):
    f = compute(n)
    return temp.format(result=f)


if __name__ == '__main__':
    main.interface.web()
