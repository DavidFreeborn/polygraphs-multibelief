"""Build the self-contained comparison report and experiment settings."""

import collections
import html
import json
import math
import statistics
from pathlib import Path
from static_charts import render_sets, shared_axis_limits

HERE=Path(__file__).resolve().parent
MODELS=['Baseline','Converging','Polarizing','Factionalizing']
NETWORKS=['Cycle','ER .1','ER .2']
ENVIRONMENTS=['Pristine','Misinformation','Disinformation']
METRICS=['Mean P(G=1)','Population SD of P(H=1)','Joint KL','Joint JS','Marginal-product JS','Within-agent belief dependence','Trust: reliable sources','Trust: unreliable sources','Source Brier','Active fraction','Exposed edge fraction','Mean P(H=1)','Population SD of P(G)']
COLOURS=['#206b91','#b17a21','#79538e','#287766','#993f50','#59666d']
STYLE='''
:root{color-scheme:light;--paper:#fbfaf7;--ink:#192d38;--muted:#58666b;--rule:#d6dbda}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:19px/1.55 Georgia,serif}main{max-width:1450px;margin:auto;padding:36px 32px 80px}h1{font-size:35px;line-height:1.2;font-weight:normal;margin:0 0 18px}h2{font-size:27px;font-weight:normal;margin:36px 0 12px}h3{font-size:22px;font-weight:normal;margin:20px 0 10px}p{max-width:1050px;margin:12px 0}a{color:#206b91;text-underline-offset:3px}nav{display:flex;gap:20px;flex-wrap:wrap;padding:14px 0;border-block:1px solid var(--rule);font:16px/1.5 system-ui,sans-serif}.intro{color:var(--muted);font-size:18px}.diagrams{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.diagram{border-bottom:1px solid var(--rule);padding-bottom:14px}.diagram svg{width:100%;height:auto;max-height:340px}.diagram p{font-size:17px;margin:4px 0}.plots{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px 18px}.plots.single{grid-template-columns:1fr}.overview{grid-template-columns:repeat(2,minmax(0,1fr))}.plot{min-width:0}.plot svg{display:block;width:100%;height:auto;background:#fff}.plot h4{font:500 16px/1.3 system-ui,sans-serif;margin:6px 0}.legend{display:flex;flex-wrap:wrap;gap:8px 22px;font:15px/1.5 system-ui,sans-serif;margin:12px 0}.legend span{display:inline-flex;align-items:center;gap:8px}.legend svg{width:32px;height:12px}.summary{border-left:2px solid #a5b0b1;padding-left:16px;font-size:18px;max-width:1120px}.collection{padding:4px 0 14px;border-bottom:1px solid var(--rule)}details{margin:14px 0}summary{cursor:pointer;line-height:1.4}summary:focus-visible,a:focus-visible{outline:3px solid #b17a21;outline-offset:4px}.group>summary{font-size:25px}.collection>summary{font-size:21px}.method{font-size:17px;color:var(--muted)}table{border-collapse:collapse;font:16px/1.45 system-ui,sans-serif;max-width:100%;display:block;overflow:auto}td,th{padding:8px 12px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}th{font-weight:600}code{font-size:15px;overflow-wrap:anywhere}pre{font:14px/1.5 monospace;overflow:auto;background:#f0f1ee;padding:18px}footer{border-top:1px solid var(--rule);margin-top:32px;padding-top:16px;font-size:16px;color:var(--muted)}.unavailable{font:15px system-ui,sans-serif;min-height:200px;display:grid;place-items:center;color:var(--muted)}
.cpt-pair{display:flex;flex-wrap:wrap;gap:18px}.cpt-pair>div{flex:1 1 220px;min-width:0}.formula{font:20px/1.9 Georgia,serif;overflow-x:auto;padding:12px 0;max-width:1100px}.formula sub,.formula sup{font-size:70%}@media(max-width:650px){.cpt-pair{display:block}}
@media(max-width:1050px){.plots{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){body{font-size:18px}main{padding:24px 14px 50px}h1{font-size:28px}.diagrams,.plots,.plots.single{grid-template-columns:1fr}.overview{grid-template-columns:1fr}.group>summary{font-size:23px}.collection>summary{font-size:20px}nav{gap:8px 16px}.diagram svg{max-height:none}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}@media print{details{display:block}summary{list-style:none}nav{display:none}.collection{break-inside:avoid}}
'''


def page(title,body,script=''):
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+title+'</title><style>'+STYLE+'</style><main>'+body+'</main>'+script+'</html>'
def percentile(values,q):
    values=sorted(values);x=(len(values)-1)*q;i=int(x)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)


def aggregate(design):
    grouped=collections.defaultdict(list)
    for path in sorted((HERE/'data/runs').glob('*.json')):
        run=json.loads(path.read_text())
        assert run['measurement_version']==3
        grouped[run['id']].append(run)
    assert len(grouped)==design['conditions']
    result={}
    for key,runs in grouped.items():
        assert len(runs)==design['repeats']
        assert {r['seed'] for r in runs}==set(design['seeds'])
        first=runs[0]
        assert all(r['times']==first['times'] for r in runs)
        fields=[]
        for metric in range(13):
            series=[]
            for t in range(len(first['times'])):
                values=[r['series'][t][metric] for r in runs if r['series'][t][metric] is not None]
                series.append([round(statistics.mean(values),8),round(percentile(values,.25),8),round(percentile(values,.75),8)] if values else [None,None,None])
            fields.append(series)
        result[key]={name:first[name] for name in ('model','topology','environment','trust_prior','times')}
        result[key]['fields']=fields
    (HERE/'summary.json').write_text(json.dumps(result,separators=(',',':')))
    return result


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+str(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def diagram(kind):
    nodes=[];edges=[]
    def node(label,x,y,observed=False):
        nodes.append(f'<rect x="{x-34}" y="{y-21}" width="68" height="42" rx="{0 if observed else 21}" fill="'+('#edf1f1' if observed else 'white')+f'" stroke="#233b48" stroke-width="1.6"/><text x="{x}" y="{y+6}" text-anchor="middle" font-size="19">{label}</text>')
    def arrow(x,y,a,b):
        length=math.hypot(a-x,b-y);u=(a-x)/length;v=(b-y)/length;xx=a-8*u;yy=b-8*v
        edges.append(f'<path d="M{x},{y}L{a},{b}" fill="none" stroke="#375463" stroke-width="1.6"/><polygon points="{a},{b} {xx-4*v},{yy+4*u} {xx+4*v},{yy-4*u}" fill="#375463"/>')
    if kind==0:
        node('H',245,65);node('Dᵢ',245,230,True);arrow(245,87,245,208)
    elif kind in (1,2):
        node('H',130,65);node('S',360,65);node('Dᵢ',245,230,True)
        arrow(145,87,227,208);arrow(345,87,263,208)
    elif kind==3:
        node('X',300,35);node('Y',440,35);node('S',370,130);node('H',120,130);node('Dᵢ',245,280,True)
        arrow(310,57,354,108);arrow(430,57,386,108);arrow(135,152,227,258);arrow(355,152,263,258)
    else:
        node('H',245,50);node('R₁',75,50);node('R₂',415,50);node('D₁,ᵢ',75,190,True);node('D₀,ᵢ',245,190,True);node('D₂,ᵢ',415,190,True)
        arrow(75,72,75,168);arrow(415,72,415,168);arrow(245,72,245,168);arrow(225,72,95,168);arrow(265,72,395,168)
        nodes.append('<text x="245" y="250" text-anchor="middle" font-size="17">D₀,ᵢ: own experiment</text>')
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 355" width="500" height="355" role="img" aria-label="'+(MODELS[kind] if kind<4 else 'Trust-in-Neighbors')+' model" font-family="Arial,sans-serif">'+''.join(edges+nodes)+'</svg>'


def model_tables(kind):
    if kind==0:
        return '<h4>D</h4>'+table(['H','P(Dᵢ=1 | H)'],[(0,'.45'),(1,'.55')])
    if kind==4:
        return '<h4>D: own observations</h4>'+table(['H','P(D₀,ᵢ=1 | H)'],[(0,'.45'),(1,'.55')])+'<h4>D: neighbour reports</h4>'+table(['H','Rⱼ','Pristine / misinformation','Disinformation'],[(0,0,'.50','.55'),(0,1,'.45','.45'),(1,0,'.50','.45'),(1,1,'.55','.55')])
    rows=[(h,ss,'.55' if (h==1 and ss==1 if kind==1 else h==ss) else '.45') for h in (0,1) for ss in (0,1)]
    d='<div><h4>D</h4>'+table(['H','S','P(Dᵢ=1 | H,S)'],rows)+'</div>'
    if kind!=3:return d
    st='<div><h4>S</h4>'+table(['X','Y','P(S=1 | X,Y)'],[(0,0,'.9'),(0,1,'.1'),(1,0,'.1'),(1,1,'.9')])+'</div>'
    return '<div class="cpt-pair">'+d+st+'</div>'


def models():
    parts=[]
    for kind,name in enumerate(MODELS+['Trust-in-Neighbors']):
        subtitle='(Bala–Goyal-like)' if kind==0 else '(O’Connor–Weatherall-like)' if kind==4 else ''
        parts.append('<section class="diagram"><h3>'+name+'</h3>'+(('<p>'+subtitle+'</p>') if subtitle else '')+diagram(kind)+model_tables(kind)+'</section>')
    return '<div class="diagrams">'+''.join(parts)+'</div>'


def label(d,dimension):
    if dimension=='model':return MODELS[d['model']]
    if dimension=='topology':return NETWORKS[d['topology']]
    if dimension=='environment':return ENVIRONMENTS[d['environment']]
    return 'Trust-in-Neighbors'


def collection(title,keys,dimension,data,trust=False,open_=False):
    spec={'keys':keys,'labels':[label(data[k],dimension) for k in keys],'metrics':[11,1,2,3,4,5]+([6,7] if trust else []),'trust':trust}
    return '<details class="collection"'+(' open' if open_ else '')+'><summary>'+html.escape(title)+'</summary><div class="chart-set" data-spec="'+html.escape(json.dumps(spec),quote=True)+'"></div></details>'


def at(d,metric,time):
    return d['fields'][metric][d['times'].index(time)][0]


def summary_tables(data):
    body='<details class="group" id="summary"><summary>Summary tables at round 500</summary>'
    for g,network in enumerate(NETWORKS):
        body+='<h3>'+network+'</h3><table><thead><tr><th rowspan="2">Belief model</th>'+''.join('<th colspan="3">'+e+'</th>' for e in ENVIRONMENTS)+'</tr><tr>'+('<th>Mean P(G=1)</th><th>Mean P(H=1)</th><th>SD P(H=1)</th>'*3)+'</tr></thead><tbody>'
        for m,q in [(0,0),(1,0),(2,0),(3,0),(4,.9)]:
            name=MODELS[m] if m<4 else 'Trust-in-Neighbors'
            body+='<tr><th>'+name+'</th>'
            for e in range(3):
                d=next(d for d in data.values() if d['model']==m and d['topology']==g and d['environment']==e and (m<4 or d['trust_prior']==(.5 if q==.5 else 1 if e==0 else .9)))
                body+=f'<td>{at(d,0,500):.3f}</td><td>{at(d,11,500):.3f}</td><td>{at(d,1,500):.3f}</td>'
            body+='</tr>'
        body+='</tbody></table>'
    return body+'</details>'


def protocol_rows(design,limits):
    return [
        ('Design','The report shows 45 settings × 100 repetitions = 4,500 runs: four belief models and one Trust-in-Neighbors variant, across three topologies and three informational environments. The .50-prior trust runs remain in the original data but are omitted from this report.'),
        ('Agents and experiments','60 agents; 500 rounds; one Bernoulli trial per active agent per round. Known action A has success probability .50; genuine B has .55. Alternatives represented by G=0 have .45.'),
        ('Topologies','Undirected cycle; connected G(60,.1); connected G(60,.2). ER graphs are rejection-sampled until connected. Each topology’s graph at a seed is reused across all model/environment settings.'),
        ('Informational environments','Pristine; six misinformants; six disinformants. Unreliable identities are sampled without replacement, fixed during a run, and shared across models/topologies at a seed.'),
        ('Private data and reports','All private B trials have success probability .55. Reliable reports equal the source’s private outcome. Misinformants send independent .50 reports; disinformants send independent .45 reports. Their private data remain genuine. One report is shared identically with all recipients. Use own data once and each neighbour report once; there is no forwarding.'),
        ('Decisions','Synchronous posterior expected-payoff maximisation. Choose B when expected success exceeds .50 with numerical tolerance 10⁻¹⁴; otherwise A. A produces no B data. Every run continues through round 500.'),
        ('Variables','H is the focal explanatory hypothesis. X,Y are the Factionalizing background nodes. Dᵢ=1 means a successful B trial; reported Dⱼ,ᵢ values are source reports. G=1 means B is strictly better than A. G is deterministic: H=1 for Baseline/trust, H=S=1 for Converging, H=S for Polarizing/Factionalizing.'),
        ('Base priors Q','For each agent and seed, independently draw root-probability parameters u,v,w from Uniform(0,1). The displayed diagrams and S table specify the base model Q before prior matching. Roots H,S have probabilities u,v in the two-node models; H,X,Y have probabilities u,v,w in Factionalizing. In Factionalizing, Q(S=1|X=Y)=.9 and otherwise .1.'),
        ('Matched initial priors P','Set q=u for the corresponding agent across all four belief models. For latent assignment z, P(z)=q Q(z|G=1) if G(z)=1, and P(z)=(1-q) Q(z|G=0) otherwise. Thus every agent has exactly the same randomly drawn P(G=1)=q across models. Q(z|G) is preserved within both strata. Matching G does not also match H or other marginal priors. Matching generally introduces dependencies, replacing independent Uniform root priors in the initial joint distribution.'),
        ('Calibration node C','C=1 is a mathematical representation of the initial prior calibration, not an experimental observation. C and the derived G indicator are suppressed in the diagrams. Let r=Q(G=1), a₁=q/r, a₀=(1-q)/(1-r), and k=1/max(a₀,a₁). Set P(C=1|G=g)=k a_g. Then Q(z|C=1) equals the matched initial prior P(z). The displayed base diagrams omit this initial conditioning; the calibrated prior need not factorise according to their root independence. No extra data are counted and C is not included in diagnostics.'),
        ('Converging change','Only H=S=1 predicts .55. The other three assignments predict .45. This replaces the preceding experiment’s two .50 alternatives. No observation identifies separate assignments within G=0.'),
        ('Trust-in-Neighbors','Own observations depend on H. Each neighbour report depends on the same H and a persistent source variable Rⱼ. Exact factorisation P(H)∏ⱼP(Rⱼ|H) is reconstructed from initial priors and cumulative counts. H’s initial prior is q=u, unchanged from the previous study.'),
        ('Initial source trust','The retained Trust-in-Neighbors model uses P(Rⱼ=1)=.90 with unreliable sources and 1.00 in pristine settings. Reliability variables start independently of each other and H. Prior 1 is dogmatic; a .90 marginal prior does not encode knowledge of an exact count.'),
        ('Source alternatives','For Rⱼ=0, report probability is .50 in pristine/misinformation settings and reversed (.55 for H=0, .45 for H=1) in disinformation settings. Agents know the alternative family. Actions and silence are conditioned on, not treated as additional observations.'),
        ('Belief statistics','Mean P(H=1) averages agents’ focal-node probabilities. Population SD uses those same probabilities, with denominator N=60. Mean P(G=1) measures belief that B is better and appears only in the cycle learning check and summary tables. All-agent values are plotted; reliable-agent statistics remain in raw records.'),
        ('Information statistics','Joint KL is the mean over ordered distinct agent pairs. Joint JS uses equal weights over unordered pairs. Marginal-product JS compares products of each agent’s individual-node marginals. Dependence is mean total correlation KL(joint || own marginal product). G is a deterministic query and C is conditioned on; both are excluded. Logs are natural and no probability smoothing is applied.'),
        ('Trust statistics','The three divergence statistics compare H alone because recipients have different source-variable sets. Full-model total correlation is ∑ⱼ I(H;Rⱼ), averaged over agents, and may depend on degree. The two trust curves average directed source-recipient edges grouped by actual source type.'),
        ('Aggregation','Each statistic is computed within a run before averaging over 100 runs. Bands are linearly interpolated run-to-run 25th–75th percentiles, not confidence intervals. Record every round 0–100, then every fifth round to 500.'),
        ('Common scales',f'Identical y-axis limits for every panel of a statistic: probabilities/trust 0–1; SD 0–.5; KL 0–{limits[2]:g} nats; JS 0–ln(2) nats; dependence 0–{limits[5]:g} nats. Limits include plotted means and quartile bands.'),
        ('Random pairing','Seeds 1729–1828. std::mt19937_64, high 53 bits mapped to Uniform[0,1). Stream offsets: +10000 topology; +20000 root priors; +30000 unreliable identities; +40000 experimental outcomes. Full private/report uniform arrays are consumed even for inactive agents.'),
        ('Execution and scope','Standalone exact C++17 runner with zlib; Python standard-library validation and reporting. Native PolyGraphs/DGL and R were not executed. Finite-horizon descriptive results, without significance claims. The calibrated initial joint is specified by the matched-prior formula; the base S table alone does not describe its conditional probabilities after calibration.')]


def measures_page():
    sections=[]
    def entry(identifier,title,summary,expression):
        sections.append('<section id="'+identifier+'"><h2>'+title+'</h2><p>'+summary+'</p><div class="formula">'+expression+'</div></section>')
    entry('mean-h','Mean P(H=1)','The average belief in H=1 across agents. It describes the population average, not the amount of disagreement.',
          'h<sub>i</sub> = P<sub>i</sub>(H=1), &nbsp; h̄ = (1/N) ∑<sub>i=1</sub><sup>N</sup> h<sub>i</sub>')
    entry('sd-h','Population SD of P(H=1)','The spread of agents’ H beliefs within one simulation. Larger values indicate greater disagreement about H.',
          'SD<sub>H</sub> = √[(1/N) ∑<sub>i=1</sub><sup>N</sup> (h<sub>i</sub> − h̄)<sup>2</sup>]')
    entry('mean-g','Mean P(G=1): P(B is better)','The average belief that B has a strictly higher success probability than A. This is a belief about the underlying success rates, not the probability of success on the next trial.',
          'g<sub>i</sub> = P<sub>i</sub>(G=1) = P<sub>i</sub>(p<sub>B</sub> &gt; p<sub>A</sub>), &nbsp; ḡ = (1/N) ∑<sub>i=1</sub><sup>N</sup> g<sub>i</sub>')
    entry('joint-kl','Joint KL','The average directed difference between agents’ joint distributions. It is sensitive to confident disagreement and has no finite upper bound.',
          'K(p ‖ q) = ∑<sub>z</sub> p(z) ln[p(z)/q(z)]<br>Joint KL = [1/(N(N−1))] ∑<sub>i≠j</sub> K(p<sub>i</sub> ‖ p<sub>j</sub>)')
    entry('joint-js','Joint JS','The average symmetric difference between agents’ joint distributions. Its range is 0 to ln(2), approximately .693 nats.',
          'J(p,q) = ½ K(p ‖ (p+q)/2) + ½ K(q ‖ (p+q)/2)<br>Joint JS = [2/(N(N−1))] ∑<sub>i&lt;j</sub> J(p<sub>i</sub>,p<sub>j</sub>)')
    entry('marginal-js','Marginal-product JS','The difference between agents’ individual-variable belief profiles, ignoring dependence within each agent’s distribution. It can increase while Joint JS decreases.',
          'q<sub>i</sub>(z₁,…,z<sub>m</sub>) = ∏<sub>k=1</sub><sup>m</sup> P<sub>i</sub>(Z<sub>k</sub>=z<sub>k</sub>)<br>Marginal-product JS = [2/(N(N−1))] ∑<sub>i&lt;j</sub> J(q<sub>i</sub>,q<sub>j</sub>)')
    entry('dependence','Within-agent belief dependence','Dependence among the variables in an individual agent’s joint distribution, averaged across agents. Each agent’s value is total correlation, which is zero exactly when its variables are mutually independent.',
          'T<sub>i</sub> = K(p<sub>i</sub> ‖ q<sub>i</sub>) = ∑<sub>z</sub> p<sub>i</sub>(z) ln[p<sub>i</sub>(z)/q<sub>i</sub>(z)]<br>Within-agent belief dependence = (1/N) ∑<sub>i=1</sub><sup>N</sup> T<sub>i</sub>')
    entry('trust-reliable','Mean trust in reliable sources','The average probability assigned to reliability on directed recipient–source links whose source is actually reliable. A source with more recipients contributes more links.',
          'Trust<sub>reliable</sub> = (1/|E<sub>+</sub>|) ∑<sub>(i,j)∈E₊</sub> P<sub>i</sub>(R<sub>j</sub>=1)')
    entry('trust-unreliable','Mean trust in unreliable sources','The corresponding average for links whose source is actually a misinformant or disinformant. It is undefined in pristine settings, where there are no unreliable sources.',
          'Trust<sub>unreliable</sub> = (1/|E<sub>−</sub>|) ∑<sub>(i,j)∈E₋</sub> P<sub>i</sub>(R<sub>j</sub>=1)')
    notation=table(['Symbol','Meaning'],[('N','Number of agents: 60.'),('pᵢ(z)','Agent i’s joint probability over the variables used in that statistic.'),('qᵢ(z)','Product of the one-variable marginals of pᵢ.'),('E₊, E₋','Directed recipient–source links with actually reliable or unreliable sources, respectively.'),('ln','Natural logarithm; information quantities are in nats.')])
    variables=table(['Model','Variables for the distribution statistics'],[('Baseline','H'),('Converging','H,S'),('Polarizing','H,S'),('Factionalizing','H,S,X,Y'),('Trust-in-Neighbors','KL, JS and marginal-product JS use H alone. Within-agent dependence uses H and all local Rⱼ variables.')])
    body='<h1>Measures</h1><nav><a href="results.html">Results</a><a href="experiment-settings.html">Experiment settings</a></nav><p>Every statistic is calculated within each simulation before averaging across the 100 runs.</p><h2>Notation</h2>'+notation+''.join(sections)+'<h2>Variables included</h2>'+variables+'<p>G is a derived indicator and is excluded from joint-distribution statistics. For Trust-in-Neighbors, H-only Joint JS and Marginal-product JS coincide; full-model total correlation is ∑ⱼ Iᵢ(H;Rⱼ), averaged over agents.</p><h2>Lines, bands and summary tables</h2><p>A line is the mean of the run-level statistic at each round. Shading is the 25th–75th percentile across runs, using linear interpolation; it is not a confidence interval.</p><p>Summary tables report the same run-averaged quantities at round 500. Their SD column averages the within-run population SD across runs; it is neither the SD of run means nor the interquartile shading.</p>'
    return page('Multibelief measures',body)


def main():
    design=json.loads((HERE/'study-design.json').read_text());validation=json.loads((HERE/'validation.json').read_text())
    assert validation['status']=='passed' and validation['runs']==5400
    data=json.loads((HERE/'summary.json').read_text()) if (HERE/'summary.json').exists() else aggregate(design)
    data={k:d for k,d in data.items() if d['model']<4 or d['trust_prior']!=.5}
    def choose(model=None,topology=None,environment=None,trust=None):
        keys=[k for k,d in data.items() if (model is None or d['model']==model) and (topology is None or d['topology']==topology) and (environment is None or d['environment']==environment) and (trust is None or (d['trust_prior']==.5)==(trust==0))]
        return sorted(keys,key=lambda k:(data[k]['model'],data[k]['topology'],data[k]['environment'],data[k]['trust_prior']))
    body='<h1>Multibelief results</h1><p class="intro">4,500 simulations · 60 agents · 500 rounds</p><nav><a href="#models">Models</a><a href="#summary">Summary tables</a><a href="#learning">P(B is better)</a><a href="#belief-comparisons">Belief models</a><a href="#environments">Environments</a><a href="#topologies">Topologies</a><a href="#trust">Trust-in-Neighbors</a><a href="measures.html">Measures</a><a href="experiment-settings.html">Experiment settings</a></nav>'
    body+='<p class="method">Lines average runs; bands show the run-to-run 25th–75th percentiles. KL, JS and within-agent belief dependence use natural logarithmic units (nats).</p><details class="group" id="models"><summary>Belief models</summary><p>G=1 means B has a strictly higher success probability than A.</p>'+models()+'</details>'
    body+=summary_tables(data)
    body+='<details class="group" id="learning" open><summary>P(B is better)</summary><h3>Cycle</h3><div class="plots">'
    for e,env in enumerate(ENVIRONMENTS):
        spec={'keys':[k for k in choose(topology=0,environment=e) if data[k]['model']<4],'labels':MODELS,'metrics':[0],'titles':[env]}
        body+='<div class="chart-set" data-spec="'+html.escape(json.dumps(spec),quote=True)+'"></div>'
    body+='</div></details><details class="group" id="belief-comparisons"><summary>Compare belief models</summary>'
    for g in range(3):
        for e in range(3):body+=collection(NETWORKS[g]+' / '+ENVIRONMENTS[e],[k for k in choose(topology=g,environment=e) if data[k]['model']<4],'model',data)
    body+='</details><details class="group" id="environments"><summary>Compare informational environments</summary>'
    for g in range(3):
        for m in range(4):body+=collection(NETWORKS[g]+' / '+MODELS[m],choose(model=m,topology=g),'environment',data)
    body+='</details><details class="group" id="topologies"><summary>Compare network topologies</summary>'
    for m in range(4):
        for e in range(3):body+=collection(MODELS[m]+' / '+ENVIRONMENTS[e],choose(model=m,environment=e),'topology',data)
    body+='</details><h2 id="trust">Trust-in-Neighbors</h2><details class="group"><summary>Compare informational environments</summary>'
    for g in range(3):
        body+=collection(NETWORKS[g],choose(model=4,topology=g),'environment',data,True)
    body+='</details><details class="group"><summary>Compare network topologies</summary>'
    for e in range(3):
        body+=collection(ENVIRONMENTS[e],choose(model=4,environment=e),'topology',data,True)
    body+='</details><footer><a href="experiment-settings.html">Experiment settings and validation</a></footer>'
    rendered=render_sets(body,data);(HERE/'results.html').write_text(page('Multibelief results',rendered))
    limits=shared_axis_limits(data)
    rows=protocol_rows(design,limits)
    settings='<h1>Experiment settings</h1><p>19 November 2026</p><nav><a href="results.html">Results</a><a href="measures.html">Measures</a><a href="#protocol">Protocol</a><a href="#validation">Validation</a></nav><h2>Models and conditional probability tables</h2>'+models()+'<h2 id="protocol">Protocol</h2>'+table(['Setting','Specification'],rows)+'<h2 id="validation">Independent validation</h2><pre>'+html.escape(json.dumps(validation,indent=2))+'</pre><h2>Replication</h2><pre>clang++ -O3 -std=c++17 worker.cpp -lz -o worker\n./worker data 500 100 all 1729\npython3 validate.py\npython3 make_report.py</pre><p>study-design.json specifies raw fields and initialisation. Per-run JSON and gzip files retain initial/final joint beliefs, graphs, private outcomes, public reports and actions. summary.json contains plotted means and quartiles. The supplied native replay script requires the full PolyGraphs fork and its DGL dependencies; it was not executed here.</p>'
    (HERE/'experiment-settings.html').write_text(page('Multibelief experiment settings',settings))
    (HERE/'experiment-settings.json').write_text(json.dumps({**design,'displayed_conditions':45,'displayed_runs':4500,'metrics':METRICS,'shared_y_axis_limits':limits,'validation':validation},indent=2))
    (HERE/'measures.html').write_text(measures_page())
    print('Built',len(data),'settings,',rendered.count('class="collection"'),'comparison collections and',rendered.count('class="plot"'),'plot panels.')


if __name__=='__main__':main()
