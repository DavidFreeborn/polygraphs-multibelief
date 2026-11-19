"""Render exact report aggregates as inline SVG, without browser dependencies."""

import html
import json
import math
import re

COLOURS=['#206b91','#b17a21','#79538e','#287766','#993f50','#59666d']
DASH=['','8 4','2 4','12 4 2 4','6 3','3 3']
METRICS=['Mean P(G=1)','Population SD of P(H=1)','Joint KL','Joint JS','Marginal-product JS','Within-agent belief dependence','Trust: reliable sources','Trust: unreliable sources','Source Brier','Active fraction','Exposed edge fraction','Mean P(H=1)','Population SD of P(G)']


def shared_axis_limits(data):
    limits={0:1,1:.5,3:math.log(2),4:math.log(2),6:1,7:1,11:1,12:.5}
    for metric in (2,5):
        maximum=max(v for d in data.values() for row in d['fields'][metric]
                    for v in row if v is not None)
        target=max(.02,maximum*1.04)
        magnitude=10**math.floor(math.log10(target))
        limits[metric]=next(step*magnitude for step in (1,1.5,2,2.5,3,4,5,6,8,10)
                            if step*magnitude>=target)
    return limits


def tick(value):
    if abs(value)<1e-10:return '0'
    return (f'{value:.3f}' if value<.1 else f'{value:.2f}' if value<10 else f'{value:.1f}' if value<100 else f'{value:.0f}').rstrip('0').rstrip('.') if value<100 else f'{value:.0f}'


def svg_plot(data,keys,metric,early=False,axis_limits=None):
    width,height,left,right,top,bottom=450,300,57,18,15,48
    xmax=100 if early else 500
    values=[v for key in keys for t,row in zip(data[key]['times'],data[key]['fields'][metric]) if t<=xmax and row[0] is not None for v in row]
    if not values:return None
    ymax=(axis_limits if axis_limits is not None else shared_axis_limits(data))[metric]
    x=lambda t:left+t/xmax*(width-left-right)
    y=lambda v:top+(height-top-bottom)*(1-v/ymax)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="450" height="300" viewBox="0 0 450 300" data-metric="{metric}" data-ymin="0" data-ymax="{ymax}" role="img"><rect width="450" height="300" fill="white"/>']
    for i in range(5):
        value=ymax*i/4;yy=y(value)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.3f}" y2="{yy:.3f}" stroke="#e3e8e8"/><text x="{left-8}" y="{yy+5:.3f}" text-anchor="end" font-family="Arial,sans-serif" font-size="14" fill="#47565c">{tick(value)}</text>')
    for i in range(6):
        value=xmax*i/5;xx=x(value)
        parts.append(f'<line x1="{xx:.3f}" x2="{xx:.3f}" y1="{top}" y2="{height-bottom}" stroke="#edf0ef"/><text x="{xx:.3f}" y="{height-bottom+24}" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#47565c">{int(value)}</text>')
    for index,key in enumerate(keys):
        points=[(t,row) for t,row in zip(data[key]['times'],data[key]['fields'][metric]) if t<=xmax and row[0] is not None]
        if not points:continue
        upper=[f'{x(t):.2f},{y(row[2]):.2f}' for t,row in points]
        lower=[f'{x(t):.2f},{y(row[1]):.2f}' for t,row in reversed(points)]
        line=' '.join(('M' if i==0 else 'L')+f'{x(t):.2f},{y(row[0]):.2f}' for i,(t,row) in enumerate(points))
        parts.append(f'<polygon points="{" ".join(upper+lower)}" fill="{COLOURS[index]}" opacity=".11"/><path d="{line}" fill="none" stroke="{COLOURS[index]}" stroke-width="2.5" stroke-dasharray="{DASH[index]}"/>')
    parts.append(f'<line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" stroke="#57696f"/><text x="{left+(width-left-right)/2}" y="{height-5}" text-anchor="middle" font-size="15" font-family="Arial,sans-serif">Round</text></svg>')
    return ''.join(parts)


def render_sets(document,data):
    axis_limits=shared_axis_limits(data)
    def replace(match):
        spec=json.loads(html.unescape(match.group(1)))
        legend=''
        if not spec.get('early'):
            legend='<div class="legend">'+''.join(f'<span><svg viewBox="0 0 32 12" aria-hidden="true"><path d="M0 6H32" stroke="{COLOURS[i]}" stroke-width="2.5" stroke-dasharray="{DASH[i]}"/></svg>{html.escape(label)}</span>' for i,label in enumerate(spec['labels']))+'</div>'
        panels=[]
        for i,metric in enumerate(spec['metrics']):
            title=spec.get('titles',METRICS)[i] if 'titles' in spec else METRICS[metric]
            svg=svg_plot(data,spec['keys'],metric,spec.get('early',False),axis_limits)
            if svg:svg=svg.replace('role="img"','role="img" aria-label="'+html.escape(title,quote=True)+'"',1)
            panels.append('<section class="plot"><h4>'+html.escape(title)+'</h4>'+(svg or '<div class="unavailable">No unreliable sources in this environment</div>')+'</section>')
        return '<div class="chart-set">'+legend+'<div class="plots'+(' single' if len(panels)==1 else '')+'">'+''.join(panels)+'</div></div>'
    return re.sub(r'<div class="chart-set" data-spec="([^"]+)"></div>',replace,document)
