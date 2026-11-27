import collections
import gzip
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
events = collections.defaultdict(list)
for path in (root/'events').glob('*.jsonl'):
    for line in path.read_text().splitlines():
        event = json.loads(line)
        events[event['request_id']].append(event)

def percentile(values, p):
    values = sorted(values)
    x = (len(values)-1)*p
    i = int(x)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)

trace = []
rows = []
for mode in ['clean','request_profile']:
    for wave in json.loads((root/(mode+'.json')).read_text()):
        group = wave['rows']
        for slot, row in enumerate(group):
            chunks = row['chunks']
            rate = int(row['headers']['x-sample-rate'])
            duration = sum(size for _,size in chunks)/(2*rate)
            latency = (row['end']-row['start'])/1e9
            first = chunks[0][0]
            played = 0
            underrun = 0
            for stamp,size in chunks:
                underrun=max(underrun,(stamp-first)/1e9-played)
                played+=size/(rate*2)
            item=dict(mode=mode,gap=wave['gap'],rep=wave['rep'],slot=slot,prompt=row['i'],audio=duration,latency=latency,rtf=latency/duration,ttfa=(first-row['start'])/1e6,underrun=underrun,wall=(max(r['end'] for r in group)-min(r['start'] for r in group))/1e9)
            if mode=='request_profile':
                e=sorted(events[row['rid']],key=lambda v:v['timestamp_ns'])
                def marker(name, stage=None):
                    return next(k for k in e if k['event_name']==name and (stage is None or k['stage']==stage))
                marks=[marker('stage_dispatch','tts_engine'),marker('scheduler_request_build_start'),marker('scheduler_queue_enter'),marker('scheduler_prefill_start'),marker('scheduler_prefill_end')]
                item['batch']=marks[-1]['metadata']['batch_size']
                for name,a,b in zip(['dispatch_to_build','build_to_queue','queue_to_prefill','prefill'],marks,marks[1:]):
                    item[name]=(b['timestamp_ns']-a['timestamp_ns'])/1e6
                    trace.append(dict(name=name,cat='scheduler',ph='X',pid=wave['gap']+1,tid=wave['rep']*4+slot,ts=a['timestamp_ns']/1000,dur=item[name]*1000,args={'batch':item['batch'],'prompt':row['i']}))
            rows.append(item)
with gzip.open(root/'scheduler-timeline.json.gz','wt') as f:json.dump({'traceEvents':trace},f)
(root/'per-request.json').write_text(json.dumps(rows,indent=2)+'\n')
for gap in [0,10,30]:
    clean=[r for r in rows if r['gap']==gap and r['mode']=='clean']
    diag=[r for r in rows if r['gap']==gap and r['mode']!='clean']
    print(gap, 'ttfa50/95',*[round(percentile([r['ttfa'] for r in clean],v),1) for v in [.5,.95]], 'rtf',round(statistics.mean(r['rtf'] for r in clean),4),'audio/s',round(sum(r['audio'] for r in clean)/(sum(r['wall'] for r in clean)/4),2),'underrun95_ms',round(1000*percentile([r['underrun'] for r in clean],.95),1),'waits',*[round(statistics.median(r[key] for r in diag),1) for key in ['dispatch_to_build','queue_to_prefill']], 'batch',dict(collections.Counter(r['batch'] for r in diag)))
