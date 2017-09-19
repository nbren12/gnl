#!/usr/bin/env python
""" Generate ipython notebooks recursively
"""
import os, glob
import re
import logging
logging.basicConfig(level=logging.INFO)

import sh
from jinja2 import Template


report = Template("""
<h2> Notebooks </h2>
<ul>
{% for link in links %}
    <li> <a href="{{ link }}"> {{ link }} </a></li>
{% endfor %}
</ul
""")


notebooks = []

if not os.path.isdir('web'):
    os.mkdir('web')
for root, dir, files in os.walk("."):

    if not re.search('.ipynb_checkpoints', root):
        for f in files:
            f = os.path.join(root, f)
            name, ext = os.path.splitext(f)
            if ext == '.ipynb':
                logging.info("Processing {}".format(name))
                htmlname = name + '.html'
                output_d_name = os.path.join('web', os.path.dirname(htmlname))
                sh.mkdir('-p', output_d_name)
                sh.jupyter('nbconvert', '--to', 'html', f, '--output', 'web/' + htmlname)

                notebooks.append(htmlname)

with open('web/index.html', 'w') as f:
    f.write(report.render(links=notebooks))
