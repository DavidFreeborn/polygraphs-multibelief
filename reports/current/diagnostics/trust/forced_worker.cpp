#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <zlib.h>
using namespace std;
#ifndef STUDY_TRIALS
#define STUDY_TRIALS 1
#endif
constexpr int N=60, TRIALS=STUDY_TRIALS;
constexpr double EPS=.05;
using Matrix=vector<vector<double>>;
double uniform01(mt19937_64& r){
    return (r()>>11)*0x1.0p-53;
}
double lse(double a,double b){
    double m=max(a,b);
    return m+log1p(exp(-abs(a-b)));
}
void normalize(vector<double>& v){
    double m=*max_element(v.begin(),v.end()),s=0;
    for(double x:v)s+=exp(x-m);
    double z=m+log(s);
    for(double&x:v)x-=z;
}
double js(const vector<double>&a,const vector<double>&b){
    double z=0;
    for(size_t k=0;k<a.size();++k){
        double m=lse(a[k],b[k])-log(2.);
        z+=(exp(a[k])*(a[k]-m)+exp(b[k])*(b[k]-m))/2;
    }
    return max(0.,z);
}
double kl_mean(const Matrix& a){
    double z=0;
    int n=a.size(),k=a[0].size();
    for(int s=0;s<k;++s){
        double p=0,l=0,pl=0;
        for(const auto&r:a){
            double q=exp(r[s]);
            p+=q;
            l+=r[s];
            pl+=q*r[s];
        }
        z+=pl/n-(p/n)*(l/n);
    }
    return max(0.,z*n/(n-1));
}
array<double,8> diagnostics(const Matrix&logs,const vector<int>&good,int bits,int focal_bit=0){
    int n=logs.size(),k=logs[0].size();
    Matrix prod(n,vector<double>(k));
    vector<double> h(n),focal(n);
    double dep=0;
    for(int i=0;i<n;++i){
        for(int s=0;s<k;++s){
            if(good[s])h[i]+=exp(logs[i][s]);
            if((s>>focal_bit)&1)focal[i]+=exp(logs[i][s]);
        }
        for(int b=0;b<bits;++b){
            double v[2]={
                0,0
            }
            ;
            for(int s=0;s<k;++s)v[(s>>b)&1]+=exp(logs[i][s]);
   // Marginals are accumulated in log space when a probability underflows.
            double ml[2];
            for(int t=0;t<2;++t){
                double m=-INFINITY;
                for(int s=0;s<k;++s)if(((s>>b)&1)==t)m=max(m,logs[i][s]);
                double z=0;
                for(int s=0;s<k;++s)if(((s>>b)&1)==t)z+=exp(logs[i][s]-m);
                ml[t]=m+log(z);
            }
            for(int s=0;s<k;++s)prod[i][s]+=ml[(s>>b)&1];
        }
        for(int s=0;s<k;++s)dep+=exp(logs[i][s])*(logs[i][s]-prod[i][s]);
    }
    double mean=accumulate(h.begin(),h.end(),0.)/n,var=0,j=0,pj=0;
    double focal_mean=accumulate(focal.begin(),focal.end(),0.)/n,focal_var=0;
    for(double x:h)var+=(x-mean)*(x-mean)/n;
    for(double x:focal)focal_var+=(x-focal_mean)*(x-focal_mean)/n;
    for(int i=0;i<n;++i)for(int q=0;q<i;++q){
        j+=js(logs[i],logs[q]);
        pj+=js(prod[i],prod[q]);
    }
    return {
        mean,sqrt(focal_var),kl_mean(logs),j*2/(n*(n-1)),pj*2/(n*(n-1)),max(0.,dep/n),focal_mean,sqrt(var)
    }
    ;
}
struct Design {
    int model,topology,environment;
    double trust;
}
;
string ident(const Design&d){
    return to_string(d.model)+"-"+to_string(d.topology)+"-"+to_string(d.environment)+"-"+to_string(d.trust);
}
vector<vector<int>> graph_for(int kind,int seed){
    mt19937_64 rng(seed+10000);
    vector<vector<int>> g(N);
    bool connected=false;
    while(!connected){
        for(auto&v:g)v.clear();
        for(int i=0;i<N;++i)for(int j=0;j<i;++j)if(kind==0?(i-j==1||i-j==N-1):uniform01(rng)<(kind==1?.1:.2)){
            g[i].push_back(j);
            g[j].push_back(i);
        }
        vector<int>seen(N);
        vector<int> todo={
            0
        }
        ;
        seen[0]=1;
        for(size_t i=0;i<todo.size();++i)for(int j:g[todo[i]])if(!seen[j]){
            seen[j]=1;
            todo.push_back(j);
        }
        connected=todo.size()==N;
    }
    return g;
}
void json_vec(ostream&f,const vector<double>&v){
    f<<'[';
    for(size_t i=0;i<v.size();++i){
        if(i)f<<',';
        if(isfinite(v[i]))f<<v[i];
        else f<<"null";
    }
    f<<']';
}
void json_mat(ostream&f,const Matrix&a){
    f<<'[';
    for(size_t i=0;i<a.size();++i){
        if(i)f<<',';
        json_vec(f,a[i]);
    }
    f<<']';
}
bool force_experiments=false;
void run(const Design&d,int seed,int rounds,const string&out){
    auto g=graph_for(d.topology,seed);
    mt19937_64 root_rng(seed+20000),bad_rng(seed+30000),trial_rng(seed+40000);
    Matrix roots(N,vector<double>(3));
    for(auto&r:roots)for(double&x:r)x=uniform01(root_rng);
    vector<int>ids(N);
    iota(ids.begin(),ids.end(),0);
    for(int i=N-1;i>0;--i){
        int j=int(uniform01(bad_rng)*(i+1));
        swap(ids[i],ids[j]);
    }
    vector<int>bad(N);
    if(d.environment)for(int i=0;i<6;++i)bad[ids[i]]=1;
    bool trust=d.model==4;
    int bits=trust||d.model==0?1:(d.model==3?4:2),k=1<<bits;
    Matrix logs(N,vector<double>(k));
    vector<double>prob(k);
    vector<int>good(k);
    for(int s=0;s<k;++s){
        int x=s&1,y=(s>>1)&1;
        prob[s]=d.model==0||trust?(x?.55:.45):d.model==1?((x&&y)?.55:.45):(x==y?.55:.45);
        good[s]=prob[s]>.500000001;
    }
    for(int i=0;i<N;++i)for(int s=0;s<k;++s){
        int x=s&1,y=(s>>1)&1,a=(s>>2)&1,b=(s>>3)&1;
        double p=x?roots[i][0]:1-roots[i][0];
        if(bits==2)p*=y?roots[i][1]:1-roots[i][1];
        if(bits==4){
            p*=(a?roots[i][1]:1-roots[i][1])*(b?roots[i][2]:1-roots[i][2]);
            double q=a==b?.9:.1;
            p*=y?q:1-q;
        }
        logs[i][s]=log(p);
    }
    if(!trust && d.model!=0){
        for(int i=0;i<N;++i){
            double mass[2]={0.,0.};
            for(int s=0;s<k;++s)mass[good[s]]+=exp(logs[i][s]);
            double q=roots[i][0];
            for(int s=0;s<k;++s)logs[i][s]+=log(good[s]?q:1-q)-log(mass[good[s]]);
            normalize(logs[i]);
        }
    }
    vector<int>sources,recipients;
    for(int i=0;i<N;++i)for(int j:g[i]){
        sources.push_back(j);
        recipients.push_back(i);
    }
    int edges=sources.size();
    vector<array<double,4>> conditional(edges);
    vector<array<int,2>> report_counts(N),private_counts(N);
    Matrix initial=logs;
    vector<double>times;
    Matrix series,ordinary;
    string name=ident(d)+"-"+to_string(seed);
    string raw=out+"/raw/"+name+".gz";
    gzFile stream=gzopen(raw.c_str(),"wb1");
    vector<int>active(N),priv(N),reports(N);
    vector<unsigned char>codes(N);
    double replay_error=0;
    auto update_trust=[&](){
        logs=initial;
        double prior0=d.trust==1?-INFINITY:log1p(-d.trust),prior1=log(d.trust);
        for(int i=0;i<N;++i)for(int h=0;h<2;++h){
            double p=h?.55:.45;
            logs[i][h]+=private_counts[i][0]*log(p)+(private_counts[i][1]-private_counts[i][0])*log1p(-p);
        }
        for(int e=0;e<edges;++e){
            int s=sources[e],i=recipients[e],n=report_counts[s][1],c=report_counts[s][0];
            for(int h=0;h<2;++h){
                double p=h?.55:.45,q=d.environment==2?1-p:.5;
                double u=prior0+c*log(q)+(n-c)*log1p(-q),v=prior1+c*log(p)+(n-c)*log1p(-p),z=lse(u,v);
                conditional[e][h*2]=u-z;
                conditional[e][h*2+1]=v-z;
                logs[i][h]+=z;
            }
        }
        for(auto&r:logs)normalize(r);
    }
    ;
    if(trust)update_trust();
    for(int t=0;t<=rounds;++t){
        if(t<=100||t%5==0||t==rounds){
            auto m=diagnostics(logs,good,bits,d.model==3?2:0);
            vector<double>row(m.begin(),m.begin()+6);
            Matrix ordinary_logs;
            for(int i=0;i<N;++i)if(!bad[i])ordinary_logs.push_back(logs[i]);
            auto om=diagnostics(ordinary_logs,good,bits,d.model==3?2:0);
            ordinary.emplace_back(om.begin(),om.begin()+6);
            double tg=0,tb=0,ng=0,nb=0,br=0,exp_edges=0;
            vector<double>tc(N);
            if(trust){
                for(int e=0;e<edges;++e){
                    int s=sources[e],i=recipients[e];
                    double r=exp(logs[i][0]+conditional[e][1])+exp(logs[i][1]+conditional[e][3]);
                    if(bad[s]){
                        tb+=r;
                        nb++;
                    }
                    else{
                        tg+=r;
                        ng++;
                    }
                    br+=(r-(bad[s]?0:1))*(r-(bad[s]?0:1));
                    exp_edges+=report_counts[s][1]>0;
                    for(int a=0;a<2;++a){
                        double lm=lse(logs[i][0]+conditional[e][a],logs[i][1]+conditional[e][2+a]);
                        for(int h=0;h<2;++h){
                            double w=exp(logs[i][h]+conditional[e][h*2+a]);
                            if(w>0)tc[i]+=w*(conditional[e][h*2+a]-lm);
                        }
                    }
                }
                row[5]=max(0.,accumulate(tc.begin(),tc.end(),0.)/N);
                double z=0,nn=0;
                for(int i=0;i<N;++i)if(!bad[i]){
                    z+=tc[i];
                    nn++;
                }
                ordinary.back()[5]=max(0.,z/nn);
            }
            double act=0;
            for(int i=0;i<N;++i){
                double p=0;
                for(int s=0;s<k;++s)p+=exp(logs[i][s])*prob[s];
                act+=p>.5+1e-14;
            }
            row.insert(row.end(),{
                trust&&ng?tg/ng:NAN,trust&&nb?tb/nb:NAN,trust?br/edges:NAN,act/N,trust?exp_edges/edges:NAN
            }
            );
            row.insert(row.end(),{m[6],m[7]});
            ordinary.back().insert(ordinary.back().end(),{NAN,NAN,NAN,NAN,NAN,om[6],om[7]});
            series.push_back(row);
            times.push_back(t);
        }
        if(t==rounds)break;
  // Consume a fixed private/report uniform array even for inactive agents.
        int genuine[N]={
        }
        ,altered[N]={
        }
        ;
        for(int channel=0;channel<2;++channel)for(int i=0;i<N;++i)for(int q=0;q<TRIALS;++q){
            double u=uniform01(trial_rng);
            if(channel==0)genuine[i]+=u<.55;
            else altered[i]+=u<(d.environment==2?.45:.5);
        }
        for(int i=0;i<N;++i){
            double p=0;
            for(int s=0;s<k;++s)p+=exp(logs[i][s])*prob[s];
            active[i]=force_experiments || p>.5+1e-14;
            priv[i]=active[i]?genuine[i]:0;
            reports[i]=active[i]?(bad[i]?altered[i]:genuine[i]):0;
            codes[i]=active[i]?priv[i]+16*reports[i]:255;
            private_counts[i][0]+=priv[i];
            private_counts[i][1]+=TRIALS*active[i];
            report_counts[i][0]+=reports[i];
            report_counts[i][1]+=TRIALS*active[i];
        }
        gzwrite(stream,codes.data(),N);
        if(trust)update_trust();
        else for(int i=0;i<N;++i){
            int c=priv[i],n=TRIALS*active[i];
            for(int j:g[i]){
                c+=reports[j];
                n+=TRIALS*active[j];
            }
            for(int s=0;s<k;++s)logs[i][s]+=c*log(prob[s])+(n-c)*log1p(-prob[s]);
            normalize(logs[i]);
        }
    }
    gzclose(stream);
    if(!trust){
        for(int i=0;i<N;++i){
            vector<double>v=initial[i];
            int c=private_counts[i][0],n=private_counts[i][1];
            for(int j:g[i]){
                c+=report_counts[j][0];
                n+=report_counts[j][1];
            }
            for(int s=0;s<k;++s)v[s]+=c*log(prob[s])+(n-c)*log1p(-prob[s]);
            normalize(v);
            for(int s=0;s<k;++s)replay_error=max(replay_error,abs(v[s]-logs[i][s]));
        }
    }
    ofstream f(out+"/runs/"+name+".json");
    f<<setprecision(16)<<"{\"measurement_version\":4,\"id\":\""<<ident(d)<<"\",\"model\":"<<d.model<<",\"topology\":"<<d.topology<<",\"environment\":"<<d.environment<<",\"trust_prior\":"<<d.trust<<",\"seed\":"<<seed<<",\"rounds\":"<<rounds<<",\"times\":";
    json_vec(f,times);
    f<<",\"series\":";
    json_mat(f,series);
    f<<",\"ordinary\":";
    json_mat(f,ordinary);
    f<<",\"roots\":";
    json_mat(f,roots);
    f<<",\"initial\":";
    json_mat(f,initial);
    f<<",\"final\":";
    json_mat(f,logs);
    f<<",\"replay_error\":"<<replay_error<<",\"edges\":[";
    for(int e=0;e<edges;++e){
        if(e)f<<',';
        f<<'['<<sources[e]<<','<<recipients[e]<<']';
    }
    f<<"],\"bad\":[";
    bool first=true;
    for(int i=0;i<N;++i)if(bad[i]){
        if(!first)f<<',';
        first=false;
        f<<i;
    }
    f<<"]}";
    if(replay_error>1e-8)throw runtime_error("Replay mismatch");
}
int main(int argc,char**argv){
    string out=argc>1?argv[1]:"data";
    int rounds=argc>2?stoi(argv[2]):500,repeats=argc>3?stoi(argv[3]):100;
    int first_seed=argc>5?stoi(argv[5]):1729;
    bool pilot=argc>4&&string(argv[4])=="pilot",trust_only=argc>4&&string(argv[4])=="trust";
    force_experiments=argc>4&&string(argv[4])=="forced";
    filesystem::create_directories(out+"/raw");
    filesystem::create_directories(out+"/runs");
    auto start=chrono::steady_clock::now();
    int done=0;
    for(int m=0;m<5;++m)for(int g=0;g<3;++g)for(int e=0;e<3;++e)for(double p:(m==4?vector<double>{
        .9
    }
    :vector<double>{
        0.
    }
    ))for(int r=0;r<repeats;++r){
        if(force_experiments && (g!=0 || e!=2 || !(m==0 || m==4)))continue;
        if(trust_only&&m!=4)continue;
        if(pilot&&(g!=2||e!=2))continue;
        run({
            m,g,e,p
        }
        ,first_seed+r,rounds,out);
        done++;
        if(done%20==0)cout<<done<<" runs, "<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<" seconds"<<endl;
    }
    cout<<"Completed "<<done<<" runs in "<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<" seconds"<<endl;
}
