#!/usr/bin/env python
"""Itinerary header generator

tags : personal
"""
from datetime import date, timedelta
import jinja2

template = """
{% for day in dates %}
## {{day.strftime("%a, %B")}} {{day.day}}
{% endfor %}
"""
jtemplate = jinja2.Template(template)

def date_range(start, end):
    d = timedelta(1)

    cur_date = start
    while cur_date <= end:
        yield cur_date
        cur_date = cur_date + d

def main():
    start_date = date(2017, 11, 29)
    end_date = date(2017, 12, 12)
    dates  = date_range(start_date, end_date)
    print(jtemplate.render(dates=dates))

if __name__ == '__main__':
    main()

