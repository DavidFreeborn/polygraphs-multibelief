"""Independent replay and distribution checks using only Python's standard library."""

import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def norm(values):
    peak = max(values)
    total = peak + math.log(sum(math.exp(x - peak) for x in values))
    return [x - total for x in values]


def logadd(a, b):
    peak = max(a, b)
    return peak + math.log1p(math.exp(-abs(a - b)))


def posterior(run, counts):
    private, reports, trials = counts
    result = []
    incoming = [[] for _ in range(60)]
    for source, recipient in run['edges']:
        incoming[recipient].append(source)
    for i in range(60):
        initial = run['initial'][i]
        values = []
        for state, prior in enumerate(initial):
            x, s = state & 1, (state >> 1) & 1
            model = run['model']
            probability = (.55 if x else .45) if model in (0, 4) else (
                (.55 if x and s else .45) if model == 1 else (.55 if x == s else .45))
            if model != 4:
                successes = private[i] + sum(reports[j] for j in incoming[i])
                total = trials[i] + sum(trials[j] for j in incoming[i])
                values.append(prior + successes * math.log(probability)
                              + (total - successes) * math.log1p(-probability))
            else:
                value = prior + private[i] * math.log(probability) + (trials[i] - private[i]) * math.log1p(-probability)
                r = run['trust_prior']
                for j in incoming[i]:
                    alternative = 1 - probability if run['environment'] == 2 else .5
                    reliable = math.log(r) + reports[j] * math.log(probability) + (trials[j] - reports[j]) * math.log1p(-probability)
                    unreliable = -math.inf if r == 1 else math.log1p(-r) + reports[j] * math.log(alternative) + (trials[j] - reports[j]) * math.log1p(-alternative)
                    value += logadd(reliable, unreliable)
                values.append(value)
        result.append(norm(values))
    return result


def metrics(logs, model):
    probabilities = [[math.exp(x) for x in row] for row in logs]
    n, k = len(logs), len(logs[0])
    bits = (k.bit_length() - 1)
    good = [bool(s & 1) if model in (0, 4) else ((s & 1) and ((s >> 1) & 1))
            if model == 1 else (s & 1) == ((s >> 1) & 1) for s in range(k)]
    gs = [sum(p[s] for s in range(k) if good[s]) for p in probabilities]
    hs = [sum(p[s] for s in range(k) if s & 1) for p in probabilities]
    mean = sum(gs) / n
    mean_h = sum(hs) / n
    products = []
    for p in probabilities:
        marginals = [[sum(p[s] for s in range(k) if ((s >> b) & 1) == a)
                      for a in (0, 1)] for b in range(bits)]
        products.append([math.prod(marginals[b][(s >> b) & 1] for b in range(bits)) for s in range(k)])
    def entropy(p):
        return -sum(x * math.log(x) for x in p if x > 0)
    def pair_js(a, b):
        return entropy([(x+y)/2 for x,y in zip(a,b)]) - (entropy(a)+entropy(b))/2
    kl = sum(sum(p[s]*(logs[i][s]-logs[j][s]) for s in range(k))
             for i,p in enumerate(probabilities) for j in range(n) if i != j) / (n*(n-1))
    js = sum(pair_js(probabilities[i],probabilities[j]) for i in range(n) for j in range(i)) * 2 / (n*(n-1))
    pjs = sum(pair_js(products[i],products[j]) for i in range(n) for j in range(i)) * 2 / (n*(n-1))
    dependence = sum(entropy(p) - entropy(q) for p,q in zip(products,probabilities)) / n
    return [mean, math.sqrt(sum((x-mean_h)**2 for x in hs)/n), kl, js, pjs, dependence,
            mean_h, math.sqrt(sum((x-mean)**2 for x in gs)/n)]


def small_joint_check():
    maximum = 0
    for prior_h in (.17,.63):
        for prior_r in (.5,.9):
            for reversed_reports in (False,True):
                weights=[]
                for h,r0,r1 in itertools.product((0,1),repeat=3):
                    p=(prior_h if h else 1-prior_h)*(prior_r if r0 else 1-prior_r)*(prior_r if r1 else 1-prior_r)
                    private_p=.55 if h else .45
                    p*=private_p**7*(1-private_p)**3
                    for r,k,n in ((r0,2,9),(r1,8,11)):
                        q=private_p if r else (1-private_p if reversed_reports else .5)
                        p*=q**k*(1-q)**(n-k)
                    weights.append(p)
                total=sum(weights)
                brute=sum(weights[4:])/total
                factored=[]
                for h in (0,1):
                    p=.55 if h else .45
                    v=(prior_h if h else 1-prior_h)*p**7*(1-p)**3
                    for k,n in ((2,9),(8,11)):
                        q=1-p if reversed_reports else .5
                        v*=prior_r*p**k*(1-p)**(n-k)+(1-prior_r)*q**k*(1-q)**(n-k)
                    factored.append(v)
                maximum=max(maximum,abs(brute-factored[1]/sum(factored)))
    return maximum


def target_states(model,k):
    return [bool(s&1) if model in (0,4) else bool(s&1) and bool(s&2) if model==1 else bool(s&1)==bool(s&2) for s in range(k)]


def initial_check(run):
    maximum=0.
    for i,actual in enumerate(run['initial']):
        u,v,w=run['roots'][i]
        k=len(actual);good=target_states(run['model'],k)
        base=[]
        for state in range(k):
            h,ss,x,y=state&1,(state>>1)&1,(state>>2)&1,(state>>3)&1
            value=u if h else 1-u
            if k==4:value*=v if ss else 1-v
            if k==16:
                value*=(v if x else 1-v)*(w if y else 1-w)
                c=.9 if x==y else .1
                value*=c if ss else 1-c
            base.append(value)
        masses=[sum(p for state,p in enumerate(base) if good[state]==g) for g in (False,True)]
        expected=[p*(u if good[state] else 1-u)/masses[good[state]] for state,p in enumerate(base)]
        maximum=max(maximum,max(abs(math.exp(a)-b) for a,b in zip(actual,expected)))
        maximum=max(maximum,abs(sum(math.exp(a) for state,a in enumerate(actual) if good[state])-u))
    return maximum


def main():
    design=json.loads((HERE/'study-design.json').read_text())
    files=sorted((HERE/'data/runs').glob('*.json'))
    errors={'initial_matched_joint':0.,'replay_log_posterior':0.,'recorded_metrics':0.,'focal_H_metrics':0.,'matched_G_series':0.,'matched_G_agent_final':0.,'baseline_log_odds':0.,'symmetric_private_only':0.,'full_joint_trust':small_joint_check(),'known_reliable_baseline':0.}
    coverage={};references={};known={};action_checks=0;previous_count=0
    sampled_seeds={1729,1779,1828}
    for count,path in enumerate(files,1):
        run=json.loads(path.read_text())
        assert run['measurement_version']==3
        assert run['rounds']==500 and all(len(row)==13 for row in run['series'])
        coverage.setdefault(run['id'],set()).add(run['seed'])
        errors['initial_matched_joint']=max(errors['initial_matched_joint'],initial_check(run))
        raw=gzip.decompress((HERE/'data/raw'/f'{path.stem}.gz').read_bytes())
        assert len(raw)==run['rounds']*60
        assert len(run['bad'])==(0 if run['environment']==0 else 6)
        bad=set(run['bad']);private=[0]*60;reports=[0]*60;trials=[0]*60
        scalar=[math.log(row[0]/(1-row[0])) for row in run['roots']]
        incoming=[[] for _ in range(60)]
        for source,recipient in run['edges']:incoming[recipient].append(source)
        for t in range(run['rounds']):
            codes=raw[t*60:(t+1)*60]
            aa=[0]*60;bb=[0]*60;nn=[0]*60
            for i,code in enumerate(codes):
                if run['model']<4 and run['seed'] in sampled_seeds:
                    q=1/(1+math.exp(-scalar[i])) if scalar[i]>-700 else 0.
                    payoff=.45+.1*q
                    assert (code!=255)==(payoff>.5+1e-14) or abs(payoff-.5)<1e-11
                    action_checks+=1
                if code!=255:
                    a,b=code&15,code>>4
                    assert 0<=a<=1 and 0<=b<=1
                    if i not in bad:assert a==b
                    aa[i]=a;bb[i]=b;nn[i]=1
                    private[i]+=a;reports[i]+=b;trials[i]+=1
            if run['model']<4 and run['seed'] in sampled_seeds:
                for i in range(60):
                    c=aa[i]+sum(bb[j] for j in incoming[i]);n=nn[i]+sum(nn[j] for j in incoming[i])
                    scalar[i]+=(2*c-n)*math.log(.55/.45)
            if run['seed'] in sampled_seeds and t+1 in (1,20,100,500):
                logs=posterior(run,(private,reports,trials))
                expected=metrics(logs,run['model']);actual=run['series'][run['times'].index(t+1)]
                for j,a in enumerate(expected):
                    if run['model']==4 and j==5:continue
                    errors['recorded_metrics']=max(errors['recorded_metrics'],abs(a-actual[j if j<6 else j+5]))
        logs=posterior(run,(private,reports,trials))
        errors['replay_log_posterior']=max(errors['replay_log_posterior'],max(abs(a-b) for row,ref in zip(logs,run['final']) for a,b in zip(row,ref)))
        hs=[sum(math.exp(lp) for state,lp in enumerate(row) if state&1) for row in logs]
        mh=sum(hs)/60;sd=math.sqrt(sum((h-mh)**2 for h in hs)/60)
        errors['focal_H_metrics']=max(errors['focal_H_metrics'],abs(mh-run['series'][-1][11]),abs(sd-run['series'][-1][1]))
        if run['model']<4:
            good=target_states(run['model'],len(logs[0]))
            finalg=[sum(math.exp(lp) for state,lp in enumerate(row) if good[state]) for row in logs]
            key=(run['topology'],run['environment'],run['seed'])
            digest=hashlib.sha256(raw).hexdigest()
            if run['model']==0:references[key]=(run['series'],digest,finalg)
            else:
                ref,expected_digest,expected_final=references[key]
                assert digest==expected_digest
                errors['matched_G_series']=max(errors['matched_G_series'],max(abs(a[j]-b[j]) for a,b in zip(ref,run['series']) for j in (0,12)))
                errors['matched_G_agent_final']=max(errors['matched_G_agent_final'],max(abs(a-b) for a,b in zip(finalg,expected_final)))
        if run['model']==0:
            for i in range(60):
                k=private[i]+sum(reports[j] for j in incoming[i]);n=trials[i]+sum(trials[j] for j in incoming[i])
                expected=run['initial'][i][1]-run['initial'][i][0]+(2*k-n)*math.log(.55/.45)
                errors['baseline_log_odds']=max(errors['baseline_log_odds'],abs(logs[i][1]-logs[i][0]-expected))
        if run['model']==4 and run['environment']==2 and run['trust_prior']==.5:
            for i in range(60):
                expected=norm([run['initial'][i][h]+private[i]*math.log(.55 if h else .45)+(trials[i]-private[i])*math.log(.45 if h else .55) for h in (0,1)])
                errors['symmetric_private_only']=max(errors['symmetric_private_only'],max(abs(math.exp(a)-math.exp(b)) for a,b in zip(logs[i],expected)))
        if run['environment']==0 and (run['model']==0 or run['model']==4 and run['trust_prior']==1):
            key=(run['topology'],run['seed'])
            if key in known:errors['known_reliable_baseline']=max(errors['known_reliable_baseline'],max(abs(a-b) for x,y in zip(known[key],run['series']) for a,b in zip(x[:5],y[:5])))
            else:known[key]=run['series']
        previous=HERE.parent/'2026-11-18-expanded-study/data/runs'/path.name
        if run['model'] in (0,4) and previous.exists():
            old=json.loads(previous.read_text());assert old['final']==run['final'] and old['series']==run['series'];previous_count+=1
        if count%540==0:print(f'Validated {count}/{len(files)}',flush=True)
    assert len(files)==5400 and len(coverage)==54
    assert all(seeds==set(design['seeds']) for seeds in coverage.values())
    assert max(errors.values())<1e-8,errors
    result={'status':'passed','runs':len(files),'conditions':54,'repeats_per_condition':100,'independent_max_absolute_errors':errors,'metric_check_seeds':sorted(sampled_seeds),'independently_checked_finite_action_decisions':action_checks,'matched_finite_evidence_and_action_histories':True,'unchanged_baseline_and_trust_runs':previous_count,'native_dgl_executed_here':False,'r_backend_executed_here':False}
    (HERE/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
