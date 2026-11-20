"""Repository evidence and expense report from historical or fresh API snapshots.

ACCEPTED reward is counted once per assignment; SUBMITTED is a separate liability.
Commission is a 40% model, not an observed bank charge. Missing rewards are unknown.
"""
import csv
import copy
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from workflow import BASE, read, save, config

ROOT=BASE.parent
OUT=ROOT/'deliverables/annotation'


def pseudonym(value):return value if str(value).startswith('worker_') else 'worker_'+hashlib.sha256(str(value).encode()).hexdigest()[:16]
def public(value, key=''):
    if key in ('token','access_token','authorization','password'):return '[REDACTED]'
    if key in ('user_id','worker_id','source_worker_id'):return pseudonym(value)
    if key in ('unavailable_for','reserved_for'):return [pseudonym(v) for v in value]
    if isinstance(value,dict):return {k:public(v,k) for k,v in value.items() if k!='owner'}
    if isinstance(value,list):return [public(v) for v in value]
    return value


def capture_time(path, value):
    if value.get('captured_at_utc'):return value['captured_at_utc']
    dates=re.findall(r'20\d{6}T\d{6}\d*Z',str(path))
    if dates:
        s=dates[-1];return datetime.strptime(s,'%Y%m%dT%H%M%S%fZ' if len(s)>16 else '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).isoformat()
    return datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()


def export():
    OUT.mkdir(parents=True,exist_ok=True)
    candidates=list((ROOT/'platform/private').rglob('*.json'))+list((BASE/'private/snapshots').rglob('*.json'))
    # A public checkout can refresh the archive from published evidence alone.
    candidates += list((OUT/'raw').glob('*.json'))
    selected={};failures=[];history=[]
    for path in candidates:
        if path.stat().st_size>50_000_000:continue
        try:value=read(path)
        except (ValueError,OSError):continue
        if not isinstance(value,dict) or not isinstance(value.get('pool'),dict) or not isinstance(value.get('assignments'),list):continue
        pid=str(value['pool'].get('id',''))
        if not pid:continue
        when=value.get('evidence_observed_at') or capture_time(path,value);history.append({'pool_id':pid,'file':str(path.relative_to(ROOT)),'captured_at':when,'assignments':len(value['assignments'])})
        if pid not in selected or when>selected[pid][0]:selected[pid]=(when,path,value)
    assignments={};pool_rows=[]
    for pid,(when,path,value) in sorted(selected.items()):
        pool=value['pool'];raw=public(value) if not path.is_relative_to(OUT/'raw') else copy.deepcopy(value)
        raw['evidence_observed_at']=when
        save(OUT/'raw'/f'pool_{pid}.json',raw)
        pool_rows.append({'pool_id':pid,'project_id':pool['project_id'],'name':pool.get('private_name',''),'status':pool.get('status'),
                          'captured_at':when,'snapshot_source':str(path.relative_to(ROOT)),
                          'url':f"https://tasks.yandex.ru/requester/project/{pool['project_id']}/pool/{pid}",
                          'price_current_snapshot':pool.get('reward_per_assignment'),'assignments':len(value['assignments'])})
        for a in value['assignments']:
            aid=a['id']
            if aid not in assignments or when>assignments[aid][0]:assignments[aid]=(when,a,pool)
    totals=defaultdict(lambda:Decimal('0'));groups=defaultdict(lambda:defaultdict(lambda:Decimal('0')));counts=Counter();unknown=[];ledger=[]
    for aid,(when,a,pool) in assignments.items():
        status=a['status'];counts[status]+=1;reward=a.get('reward');name=pool.get('private_name','')
        stage='training' if pool.get('type')=='TRAINING' or 'training' in name else 'exam' if 'exam' in name else 'rework' if 'rework' in name else 'review' if ':review_' in name else 'bbox'
        if reward is None and stage=='training':reward=0
        if reward is None and status in ('ACCEPTED','SUBMITTED'):unknown.append(aid)
        amount=Decimal(str(reward)) if reward is not None else None
        if amount is not None and status in ('ACCEPTED','SUBMITTED'):
            bucket='accepted' if status=='ACCEPTED' else 'pending';totals[bucket]+=amount;groups[stage][bucket]+=amount
        ledger.append({'assignment_id':aid,'pool_id':a['pool_id'],'worker_id':pseudonym(a['user_id']),'status':status,'reward':reward,'stage':stage,'snapshot_time':when})
    rate=Decimal(str(config()['commission_rate']));accepted=totals['accepted'];pending=totals['pending']
    budget={'accepted_rewards_rub':float(accepted),'accepted_plus_commission_estimate_rub':float((accepted*(1+rate)).quantize(Decimal('.01'))),
            'pending_rewards_reserve_rub':float(pending),'accepted_plus_pending_rub':float(accepted+pending),
            'accepted_plus_pending_plus_commission_estimate_rub':float(((accepted+pending)*(1+rate)).quantize(Decimal('.01'))),
            'commission_rate_assumed':float(rate),'unknown_reward_assignment_ids':unknown,
            'status_counts':dict(counts),'by_stage':{k:{s:float(v) for s,v in vals.items()} for k,vals in groups.items()},
            'coverage':'Historical local snapshots, latest observation per pool/assignment; refresh via notebook 03 for current status.',
            'limitations':'Not a billing statement. Bonuses, taxes/refunds/transactions not included. Pending is a reserve, not an expense. Missing reward is unknown, never replaced with current pool price.'}
    save(OUT/'budget.json',budget);save(OUT/'pool_inventory.json',pool_rows);save(OUT/'snapshot_history.json',history)
    for name,rows in [('assignment_ledger.csv',ledger),('pool_inventory.csv',pool_rows)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['empty']);writer.writeheader();writer.writerows(rows)
    active_path=ROOT/'platform/private/active_setup.json'
    active=read(active_path) if active_path.exists() else None
    base=(ROOT/active['registry']).parent if active else ROOT/'platform/private/unavailable'
    for rel in ['plan.json','reviewed_rework/manual_decisions.json','current_dataset_quality/latest.json','final_export/provenance.json','current_dataset_quality/user_final_decisions.json']:
        path=base/rel
        if path.exists():save(OUT/'decisions'/Path(rel).name,public(read(path)))
    for name in ['butterfly-box.html','butterfly-review.html']:
        shutil.copyfile(ROOT/name,OUT/name)
    for name in ['final_corrections.json','final_corrections_mapping.json']:
        shutil.copyfile(ROOT/'data/qualification'/name,OUT/name)
    shutil.copyfile(ROOT/'data/manifests/dataset_v1.jsonl',OUT/'dataset_manifest.jsonl')
    final=ROOT/'exports/final_butterflies_v1'
    save(OUT/'dataset_statistics.json',read(final/'validation_report.json'))
    for role in ('bbox','review'):
        shutil.copyfile(BASE/'assets'/f'{role}_project.json',OUT/f'{role}_project_template.json')
    save(OUT/'clean_config.json',config())
    registry=read(BASE/'private/existing_registry.json') if (BASE/'private/existing_registry.json').exists() else {'objects':read(OUT/'existing_resources.json')['objects']}
    save(OUT/'existing_resources.json',{'objects':{k:{'id':v['id'],'resource':v['resource']} for k,v in registry['objects'].items()}})
    (OUT/'README.md').unlink(missing_ok=True)
    hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'};save(OUT/'SHA256SUMS.json',hashes)
    with ZipFile(OUT.parent/'annotation_evidence.zip','w',ZIP_DEFLATED) as archive:
        for p in OUT.rglob('*'):
            if p.is_file():archive.write(p,str(p.relative_to(OUT.parent)))
    print(json.dumps(budget,ensure_ascii=False,indent=2));print('Evidence pools:',len(selected),'assignments:',len(assignments));return budget

if __name__=='__main__':export()
