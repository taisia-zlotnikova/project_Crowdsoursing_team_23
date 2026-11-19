"""Clean annotation workflow. Standard library only. No imports from legacy scripts.

Each remote mutation requires Client(write=True). Paid issuing is a separate launch().
Tokens exist in memory only. Private journal is persisted before each creation/upload.
"""
import copy
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
API = 'https://tasks.yandex.ru/api/v1'
SKILLS = ('bbox_exam_score','bbox_access','review_exam_score','review_control_score')


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def read(path): return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def now():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


class Client:
    def __init__(self, token, write=False):
        self._token=token;self.write=write
    def request(self, method, path, body=None, params=None):
        if method!='GET' and not self.write:raise PermissionError('Read-only client. Enable write=True explicitly.')
        base='https://tasks.yandex.ru' if path.startswith('/api/new/') else API
        url=base+path+('?' + urllib.parse.urlencode(params) if params else '')
        req=urllib.request.Request(url,method=method,data=None if body is None else json.dumps(body).encode(),headers={'Authorization':'OAuth '+self._token,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=60) as response:
                data=response.read();return json.loads(data) if data else None
        except urllib.error.HTTPError as exc:
            detail=exc.read().decode(errors='replace').replace(self._token,'[REDACTED]')
            raise RuntimeError(f'HTTP {exc.code}: {detail[:2000]}; no automatic write retry') from None
        except (urllib.error.URLError,TimeoutError,OSError):
            raise RuntimeError('Network failure; a write may have succeeded. Reconcile journal before retrying.') from None
    def all(self, resource, **params):
        rows=[];query={**params,'sort':'id','limit':100}
        while True:
            page=self.request('GET','/'+resource,params=query);items=page['items'];rows+=items
            if not page.get('has_more'):return rows
            if not items or query.get('id_gt')==items[-1]['id']:raise RuntimeError('Pagination did not advance')
            query['id_gt']=items[-1]['id']


def config():
    c=read(BASE/'config.json')
    for key,s in c['stages'].items():
        if not isinstance(s['price'],(int,float)) or isinstance(s['price'],bool) or not math.isfinite(s['price']) or s['price']<=0:raise ValueError(key+': invalid price')
        for f in ('page_size','overlap'):
            if not isinstance(s[f],int) or isinstance(s[f],bool) or s[f]<1:raise ValueError(key+': invalid '+f)
        if s['deferred_days'] not in range(0,22):raise ValueError('Invalid acceptance period')
    for key in ('bbox_exam','review_exam'):
        if c['stages'][key]['page_size']!=10:raise ValueError('Exams must contain all ten examples on ONE page')
    return c


def registry_path(profile):
    if profile not in ('existing','new'):raise ValueError('profile must be existing or new')
    return BASE/'private'/('existing_registry.json' if profile=='existing' else 'new_registry.json')


def state_for(client, profile):
    p=registry_path(profile);owner=str(client.request('GET','/requester')['id'])
    state=read(p) if p.exists() else {'account_id':owner,'objects':{}}
    if state['account_id']!=owner:raise ValueError('Wrong requester account')
    return state,p


def ids_for(state):return {k:str(v['id']) for k,v in state['objects'].items() if v.get('id')}
def subset(actual, expected):
    if isinstance(expected,dict):return isinstance(actual,dict) and all(k in actual and subset(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):return isinstance(actual,list) and actual==expected
    return actual==expected


def ensure(client, state, path, key, resource, body):
    obj=state['objects'].get(key,{})
    if obj.get('id'):return str(obj['id'])
    marker='name' if resource=='skills' else 'private_comment' if resource=='projects' else 'private_name'
    matches=[r for r in client.all(resource,**({'project_id':body['project_id']} if resource in ('pools','trainings') else {})) if r.get(marker)==body[marker]]
    if len(matches)>1:raise RuntimeError('Duplicate remote marker: '+key)
    if matches:remote=matches[0]
    elif obj.get('pending'):raise RuntimeError('Uncertain creation: '+key+'. Inspect remote account; no blind retry.')
    else:
        state['objects'][key]={'resource':resource,'pending':True,'body_hash':digest(body)};save(path,state)
        remote=client.request('POST','/'+resource,body)
    state['objects'][key]={'resource':resource,'id':str(remote['id']),'pending':False,'body_hash':digest(body)};save(path,state)
    return str(remote['id'])


def mixer(real,gold):return {'real_tasks_count':real,'golden_tasks_count':gold,'training_tasks_count':0,'min_real_tasks_count':real,'min_golden_tasks_count':gold,'min_training_tasks_count':0,'force_last_assignment':False,'shuffle_tasks_in_task_suite':True}
def skill(key,op='EQ',value=None):return {'category':'skill','key':key,'operator':op,**({'value':value} if value is not None else {})}


def pool_body(c, key, ids):
    role=key.split('_')[0];s=c['stages'][key];exam=key.endswith('_exam') and key in ('bbox_exam','review_exam')
    review=role=='review';configs=[]
    if exam:
        filt={'and':[skill(ids['review_exam_score'] if review else ids['bbox_access'])]}
        configs=[{'collector_config':{'type':'ANSWER_COUNT'},'rules':[{'conditions':[{'key':'assignments_accepted_count','operator':'GTE','value':1}],'action':{'type':'RESTRICTION_V2','parameters':{'scope':'POOL','duration_unit':'PERMANENT','private_comment':'Экзамен завершён.'}}}]}]
    elif review:
        filt={'and':[skill(ids['review_exam_score'],'GTE',c['review_exam_pass']),{'or':[skill(ids['review_control_score']),skill(ids['review_control_score'],'GTE',c['review_control_pass'])]}]}
    else:filt={'and':[skill(ids['bbox_exam_score'],'GTE',c['bbox_pass']),skill(ids['bbox_access'],'EQ',100)]}
    if review:
        score=ids['review_exam_score'] if exam else ids['review_control_score'];minimum=10 if exam else 5
        enough={'key':'golden_set_answers_count','operator':'GTE','value':minimum}
        rules=[{'conditions':[enough],'action':{'type':'SET_SKILL_FROM_OUTPUT_FIELD','parameters':{'skill_id':score,'from_field':'golden_set_correct_answers_rate'}}}]
        if not exam:rules.append({'conditions':[enough,{'key':'golden_set_correct_answers_rate','operator':'LT','value':c['review_control_pass']}],'action':{'type':'RESTRICTION_V2','parameters':{'scope':'POOL','duration_unit':'PERMANENT','private_comment':'Низкое качество контрольных; требуется разбор.'}}})
        configs.append({'collector_config':{'type':'GOLDEN_SET','parameters':{'history_size':10}},'rules':rules})
    qc={'configs':configs}
    if exam:qc['training_requirement']={'training_pool_id':ids[role+'_training'],'training_passing_skill_value':c['training_pass']}
    mix=mixer(0,10) if review and exam else mixer(s['page_size'],1 if review else 0)
    if not exam:mix.update(min_real_tasks_count=1,force_last_assignment=True,force_last_assignment_delay_seconds=60)
    return {'project_id':ids[role],'private_name':c['namespace']+':'+key,'private_comment':'Clean workflow: '+key,'will_expire':c['expires_at'],'may_contain_adult_content':False,'reward_per_assignment':s['price'],'assignment_max_duration_seconds':3600 if exam else 1500,'auto_accept_solutions':s['deferred_days']==0,'auto_accept_period_day':max(1,s['deferred_days']),'defaults':{'default_overlap_for_new_tasks':s['overlap'],'default_overlap_for_new_task_suites':s['overlap']},'quality_control':qc,'filter':filt,'mixer_config':mix}


def prepare(client, profile='new'):
    c=config();state,path=state_for(client,profile)
    if profile=='existing':raise ValueError('Existing resources already exist. Use inventory/update_settings; preparation is new-profile only.')
    for key in SKILLS:ensure(client,state,path,key,'skills',{'name':c['namespace']+':'+key,'hidden':True})
    for role in ('bbox','review'):
        body=read(BASE/'assets'/f'{role}_project.json');body['private_comment']=c['namespace']+':'+role;body['public_instructions']=(BASE/c['instructions'][role]).read_text()
        ensure(client,state,path,role,'projects',body)
    for role,n in [('bbox',5),('review',6)]:
        ids=ids_for(state);ensure(client,state,path,role+'_training','trainings',{'project_id':ids[role],'private_name':c['namespace']+':'+role+'_training','assignment_max_duration_seconds':1800,'may_contain_adult_content':False,'mix_tasks_in_creation_order':True,'shuffle_tasks_in_task_suite':True,'training_tasks_in_task_suite_count':n,'task_suites_required_to_pass':1,'retry_training_after_days':7,'inherited_instructions':True})
    ids=ids_for(state)
    for key in c['stages']:ensure(client,state,path,key,'pools',pool_body(c,key,ids))
    sources=read(BASE/'assets/task_sources.json')
    for key in ('bbox_training','review_training','bbox_exam','review_exam','bbox_pilot','bbox_main'):
        upload(client,profile,key,sources[key])
    for key in ('review_bbox_exam','review_pilot','review_main'):upload(client,profile,key,sources['review_controls'])
    return inventory(client,profile)


def upload(client, profile, key, tasks):
    state,path=state_for(client,profile);ids=ids_for(state);pid=ids[key];resource='trainings' if key.endswith('_training') else 'pools'
    pool=client.request('GET','/'+resource+'/'+pid)
    if pool['status'] not in ('OPEN','CLOSED'):raise RuntimeError('Pool unavailable')
    existing={t['input_values']['item_key']:t for t in client.all('tasks',pool_id=pid)}
    prepared=[]
    for source in tasks:
        task=copy.deepcopy(source);task.pop('id',None);task['pool_id']=pid
        if task.get('known_solutions') and key in ('bbox_training','review_training'):
            task['infinite_overlap']=True;task.pop('overlap',None)
        elif task.get('known_solutions') and key.startswith('review_') and key!='review_exam':
            task['infinite_overlap']=True;task.pop('overlap',None)
        else:task['overlap']=config()['stages'][key]['overlap'];task['infinite_overlap']=False
        old=existing.get(task['input_values']['item_key'])
        if old:
            for field in ('input_values','known_solutions'):
                if field in task and not subset(old.get(field),task[field]):raise RuntimeError('Existing task differs: '+task['input_values']['item_key'])
            continue
        prepared.append(task)
    journal=path.parent/'uploads'/f'{key}.json'
    previous=read(journal) if journal.exists() else {}
    if previous.get('status')=='pending' and any(k not in existing for k in previous['keys']):raise RuntimeError('Uncertain upload; reconcile journal before retrying')
    for start in range(0,len(prepared),50):
        batch=prepared[start:start+50];save(journal,{'status':'pending','keys':[t['input_values']['item_key'] for t in batch]})
        client.request('POST','/tasks',batch,{'open_pool':'false','async_mode':'false','skip_invalid_items':'false','allow_defaults':'false'})
        actual={t['input_values']['item_key']:t for t in client.all('tasks',pool_id=pid)}
        if any(not subset(actual.get(t['input_values']['item_key']),{'input_values':t['input_values']}) for t in batch):raise RuntimeError('Upload verification failed')
        save(journal,{'status':'verified','keys':[t['input_values']['item_key'] for t in batch]})
    return {'pool':key,'added':len(prepared),'already_present':len(tasks)-len(prepared)}


def inventory(client,profile='existing'):
    state,_=state_for(client,profile);rows=[]
    for key,obj in state['objects'].items():
        if obj['resource'] not in ('pools','trainings'):continue
        r=client.request('GET','/'+obj['resource']+'/'+str(obj['id']))
        rows.append({'key':key,'id':obj['id'],'status':r['status'],'price':r.get('reward_per_assignment',0),'mixer':r.get('mixer_config'),'filter':r.get('filter'),'quality_control':r.get('quality_control'),'url':f"https://tasks.yandex.ru/requester/project/{r['project_id']}/pool/{obj['id']}"})
    save(BASE/'private'/f'{profile}_inventory.json',rows);return rows


def transition(client, resource, pid, action):
    current=client.request('GET',f'/{resource}/{pid}');desired='OPEN' if action=='open' else 'CLOSED'
    if current['status']==desired:return current
    operation=client.request('POST',f'/{resource}/{pid}/{action}')
    for _ in range(45):
        current=client.request('GET',f'/{resource}/{pid}')
        if current['status']==desired:return current
        if operation and operation.get('status')=='FAIL':raise RuntimeError('Lifecycle operation failed')
        if operation and operation.get('id'):operation=client.request('GET','/operations/'+operation['id'])
        time.sleep(1)
    raise RuntimeError('Operation pending; inspect status before retrying')


def launch(client,profile,key):
    state,_=state_for(client,profile);ids=ids_for(state)
    if key.endswith('_training'):return transition(client,'trainings',ids[key],'open')
    pool=client.request('GET','/pools/'+ids[key])
    if pool['status']=='OPEN':return {'status':'OPEN','id':ids[key]}
    if not client.all('tasks',pool_id=ids[key]):raise ValueError('Cannot launch an empty pool')
    training=pool.get('quality_control',{}).get('training_requirement',{}).get('training_pool_id')
    if training:transition(client,'trainings',training,'open')
    endpoint='/api/new/requester/poolModeration/'+ids[key]
    try:moderation=client.request('GET',endpoint)
    except RuntimeError as exc:
        if 'HTTP 404' not in str(exc):raise
        moderation={}
    status=moderation.get('status')
    if status=='BLOCKED':raise RuntimeError('Moderation blocked; inspect account')
    if status not in ('PASSED','READY','IN_PROGRESS'):status=client.request('PUT',endpoint,{'status':'READY'})['status']
    if status!='PASSED':return {'status':'WAITING_MODERATION','id':ids[key]}
    return transition(client,'pools',ids[key],'open')


def update_settings(client,profile,key,**changes):
    # Existing tasks require explicit overlap updates; never change page size after issuing.
    allowed={'price','overlap','page_size'}
    if not changes or set(changes)-allowed:raise ValueError('Allowed: price, overlap, page_size')
    state,_=state_for(client,profile);pid=ids_for(state)[key];before=client.request('GET','/pools/'+pid)
    if 'page_size' in changes and client.all('assignments',pool_id=pid):raise ValueError('Page size cannot change after issuing; create another cohort')
    for k,v in changes.items():
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:raise ValueError('Invalid '+k)
        if k!='price' and not isinstance(v,int):raise ValueError(k+' must be integer')
    writable=set(pool_body(config(),key,ids_for(state)))
    body={k:copy.deepcopy(v) for k,v in before.items() if k in writable}
    if 'price' in changes:body['reward_per_assignment']=changes['price']
    if 'overlap' in changes:body['defaults']={'default_overlap_for_new_tasks':changes['overlap'],'default_overlap_for_new_task_suites':changes['overlap']}
    if 'page_size' in changes:
        f='golden_tasks_count' if key=='review_exam' else 'real_tasks_count';body['mixer_config'][f]=changes['page_size'];body['mixer_config']['min_'+f]=changes['page_size']
    save(BASE/'private/changes'/f'{now()}_{key}_before.json',before)
    was_open=before['status']=='OPEN'
    if was_open:transition(client,'pools',pid,'close')
    after=client.request('PUT','/pools/'+pid,body)
    if not subset(client.request('GET','/pools/'+pid),body):raise RuntimeError('Settings not verified; pool remains closed')
    if 'overlap' in changes:
        for task in client.all('tasks',pool_id=pid):
            if not task.get('infinite_overlap'):client.request('PATCH','/tasks/'+task['id'],{'overlap':changes['overlap'],'infinite_overlap':False})
    if was_open:transition(client,'pools',pid,'open')
    return {'key':key,'changes':changes,'status':'OPEN' if was_open else 'CLOSED'}


def snapshot(client,profile='existing'):
    state,_=state_for(client,profile);folder=BASE/'private/snapshots'/now();folder.mkdir(parents=True)
    state=copy.deepcopy(state)
    if profile=='existing':
        historical=BASE.parent/'deliverables/annotation/pool_inventory.json'
        if historical.exists():
            known={str(v['id']) for v in state['objects'].values()}
            for pool in read(historical):
                if str(pool['pool_id']) not in known:
                    name=pool['name'].rsplit(':',1)[-1]
                    state['objects']['historical_'+name+'_'+str(pool['pool_id'])]={'id':str(pool['pool_id']),'resource':'trainings' if 'training' in name else 'pools'}
    errors=[]
    for key,obj in state['objects'].items():
        if obj['resource'] not in ('pools','trainings'):continue
        try:
            pool=client.request('GET','/'+obj['resource']+'/'+str(obj['id']))
            tasks=client.all('tasks',pool_id=obj['id']);answers=client.all('assignments',pool_id=obj['id'])
            save(folder/(key+'_raw.json'),{'captured_at_utc':datetime.now(timezone.utc).isoformat(),'pool':pool,'tasks':tasks,'assignments':answers})
        except Exception as exc:errors.append({'key':key,'error':str(exc)})
    save(folder/'errors.json',errors)
    if errors:raise RuntimeError('Incomplete snapshot; inspect '+str(folder/'errors.json'))
    save(BASE/'private/latest_snapshot.json',{'directory':str(folder.relative_to(BASE))});return folder


def output_boxes(output):
    status=output.get('status');boxes=output.get('result',[])
    if status not in ('HAS_BUTTERFLY','NO_BUTTERFLY'):raise ValueError('Ambiguous or technical answer needs manual inspection')
    if (status=='HAS_BUTTERFLY')!=bool(boxes):raise ValueError('Status/boxes conflict')
    for b in boxes:
        if b.get('shape')!='rectangle':raise ValueError('Unsupported shape')
        x,y,w,h=(b[k] for k in ('left','top','width','height'))
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in (x,y,w,h)) or min(x,y)<0 or min(w,h)<=0 or x+w>1+1e-8 or y+h>1+1e-8:raise ValueError('Invalid normalized rectangle')
    return boxes


def handoff(client,profile,source_key,target_key):
    state,path=state_for(client,profile);ids=ids_for(state);source=ids[source_key];target=ids[target_key]
    assignments=client.all('assignments',pool_id=source);authors={a['user_id'] for a in assignments}
    # New version is keyed by assignment/task, never just photo_id.
    tasks=[];manual=[]
    for a in assignments:
        if a['status'] not in ('SUBMITTED','ACCEPTED'):continue
        if len(a['tasks'])!=len(a['solutions']):raise ValueError('Tasks/solutions mismatch')
        for t,s in zip(a['tasks'],a['solutions']):
            iv=t['input_values'];out=s['output_values'];aid=f"bbox_{a['id']}_{t['id']}_v1"
            try:boxes=output_boxes(out)
            except ValueError as exc:manual.append({'annotation_id':aid,'photo_id':iv['photo_id'],'reason':str(exc),'output_values':out});continue
            task={'pool_id':target,'overlap':config()['stages'][target_key]['overlap'],'infinite_overlap':False,'unavailable_for':sorted(authors),
                  'input_values':{'image':iv['image'],'photo_id':iv['photo_id'],'item_key':aid,'annotation_id':aid,'source_assignment_id':a['id'],'source_task_id':t['id'],'source_status':out['status'],'result':boxes}}
            tasks.append(task)
    save(path.parent/'handoffs'/f'{source_key}_manual.json',manual)
    result=upload(client,profile,target_key,tasks)
    save(path.parent/'handoffs'/f'{source_key}_result.json',result)
    return {'upload':result,'manual':len(manual)}


def review_decisions(folder):
    """Pure report over a complete snapshot. Controls excluded; self-votes excluded."""
    files=list(Path(folder).glob('*_raw.json'));sources=[];votes=defaultdict(dict)
    for file in files:
        raw=read(file);is_bbox=file.name.startswith('bbox')
        for a in raw['assignments']:
            if a['status'] not in ('SUBMITTED','ACCEPTED'):continue
            if len(a['tasks'])!=len(a['solutions']):raise ValueError('Misaligned answers')
            for t,s in zip(a['tasks'],a['solutions']):
                if t.get('known_solutions'):continue
                iv=t['input_values'];out=s.get('output_values',{})
                if is_bbox:
                    if iv.get('mode')=='bbox':sources.append({'photo_id':str(iv['photo_id']),'annotation_id':f"bbox_{a['id']}_{t['id']}_v1",'worker_id':a['user_id'],'assignment_id':a['id'],'task_id':t['id'],'input_values':iv,'output_values':out,'submitted':a.get('submitted',''),'stage':file.stem.replace('_raw','')})
                elif iv.get('annotation_id') and not iv['annotation_id'].startswith('service:'):
                    votes[iv['annotation_id']][a['user_id']]={'verdict':out.get('verdict'),'input_values':iv}
    rows=[]
    for source in sources:
        counts=Counter()
        for worker,v in votes[source['annotation_id']].items():
            iv=v['input_values']
            if worker==source['worker_id'] or iv.get('source_assignment_id')!=source['assignment_id'] or iv.get('source_task_id')!=source['task_id'] or iv.get('result')!=source['output_values'].get('result',[]) or iv.get('source_status')!=source['output_values'].get('status'):continue
            if v['verdict'] in ('OK','BAD','UNSURE'):counts[v['verdict']]+=1
        try:output_boxes(source['output_values']);technical=False
        except ValueError:technical=True
        rows.append({**source,'votes':dict(counts),'reviews':sum(counts.values()),'decision':'MANUAL' if technical else 'REWORK' if counts['BAD']>counts['OK'] else 'OK' if sum(counts.values())>=3 and counts['OK']>counts['BAD'] else 'PENDING'})
    save(Path(folder)/'review_decisions.json',rows);return rows


def rework(client,profile,folder,manual_file=None):
    rows=review_decisions(folder);manual=read(manual_file) if manual_file else read(BASE/'assets/manual_by_photo.json')
    # Latest submitted version only. Prefer a rework stage over a historical main answer.
    latest={}
    for row in rows:
        if row['stage']=='bbox_exam':continue
        old=latest.get(row['photo_id'])
        if not old or (row['stage']=='bbox_rework',row['submitted'])>(old['stage']=='bbox_rework',old['submitted']):latest[row['photo_id']]=row
    tasks=[]
    for photo,row in latest.items():
        decision=manual.get(photo,{}).get('verdict')
        if decision=='GOOD' or row['decision']!='REWORK':continue
        if row['stage']=='bbox_rework':raise ValueError('Another rework round needs a separate stage; do not overwrite completed tasks')
        iv=copy.deepcopy(row['input_values']);iv['item_key']='rework:'+row['annotation_id']
        tasks.append({'input_values':iv,'unavailable_for':[row['worker_id']]})
    return upload(client,profile,'bbox_rework',tasks)


def admit(client,profile,worker_ids,source='review',scores=None):
    """BBOX admission only; source='manual' requires an explicit author decision."""
    state,path=state_for(client,profile);ids=ids_for(state);c=config();scores=scores or {}
    receipt=[]
    for worker in worker_ids:
        score=scores.get(worker)
        if source=='manual':score=c['bbox_pass'] if score is None else score
        if score is None or score<c['bbox_pass']:raise ValueError('No passing quality score for '+worker)
    for worker in worker_ids:
        score=scores.get(worker,c['bbox_pass'])
        for name,value in [('bbox_exam_score',score),('bbox_access',100)]:
            client.request('PUT','/user-skills',{'skill_id':ids[name],'user_id':worker,'value':value})
            actual=client.all('user-skills',skill_id=ids[name],user_id=worker)
            if len(actual)!=1 or actual[0]['value']!=value:raise RuntimeError('Skill write not verified')
        receipt.append({'worker_id':worker,'score':score,'source':source})
        save(path.parent/'admission'/f'{profile}_latest.json',receipt)
    return receipt


def grade_exam(folder,pass_score=None):
    rows=[r for r in review_decisions(folder) if r['stage']=='bbox_exam'];grouped=defaultdict(list)
    for r in rows:grouped[r['worker_id']].append(r)
    result=[]
    for worker,answers in grouped.items():
        complete=len(answers)==10 and len({a['photo_id'] for a in answers})==10 and all(a['reviews']>=3 and a['decision'] in ('OK','REWORK') for a in answers)
        score=10*sum(a['decision']=='OK' for a in answers) if complete else None
        result.append({'worker_id':worker,'complete':complete,'score':score,'pass':complete and score>=(pass_score or config()['bbox_pass'])})
    return result


def attach_existing(client):
    """Attach the documented historical pool IDs without mutating the platform."""
    mapping=read(BASE.parent/'deliverables/annotation/existing_resources.json')
    state={'account_id':str(client.request('GET','/requester')['id']),'objects':mapping['objects']}
    # Check ownership of both projects, not just whether IDs are syntactically valid.
    for role in ('bbox','review'):
        project=client.request('GET','/projects/'+state['objects'][role]['id'])
        if not project.get('owner',{}).get('myself',False):raise ValueError('Existing projects do not belong to this requester')
    save(registry_path('existing'),state);return state


def plan_costs():
    c=config();sources=read(BASE/'assets/task_sources.json');rows=[]
    counts={'bbox_exam':10,'review_exam':10,'bbox_pilot':len(sources['bbox_pilot']),'bbox_main':len(sources['bbox_main'])}
    for key,s in c['stages'].items():
        count=counts.get(key)
        if key in ('review_bbox_exam','review_pilot','review_main'):
            count={'review_bbox_exam':10*c['stages']['bbox_exam']['overlap'],'review_pilot':25,'review_main':375}[key]
        pages=math.ceil(count*s['overlap']/s['page_size']) if count is not None else None
        rows.append({'stage':key,'real_inputs':count,'nominal_pages':pages,'price_per_page':s['price'],
                     'reward_estimate':round(pages*s['price'],2) if pages is not None else None,
                     'with_commission_estimate':round(pages*s['price']*(1+c['commission_rate']),2) if pages is not None else None})
    return {'stages':rows,'note':'Nominal full-page estimate, not a guarantee. Partial pages, repeats and rework cost extra. Training is unpaid.'}


def update_quality(client,profile):
    """Apply configured REVIEW thresholds without changing prices or schemas."""
    c=config();state,_=state_for(client,profile);ids=ids_for(state);result=[]
    for key in ('review_bbox_exam','review_pilot','review_main'):
        pid=ids[key];before=client.request('GET','/pools/'+pid)
        if before['status'] not in ('OPEN','CLOSED'):raise ValueError('Pool unavailable: '+key)
        fields=set(pool_body(c,key,ids));body={k:copy.deepcopy(v) for k,v in before.items() if k in fields}
        body['filter']=pool_body(c,key,ids)['filter']
        for collector in body['quality_control']['configs']:
            if collector['collector_config']['type']!='GOLDEN_SET':continue
            for rule in collector['rules']:
                for condition in rule['conditions']:
                    if condition['key']=='golden_set_correct_answers_rate' and condition['operator']=='LT':condition['value']=c['review_control_pass']
        save(BASE/'private/changes'/f'{now()}_{key}_quality_before.json',before)
        was_open=before['status']=='OPEN'
        if was_open:transition(client,'pools',pid,'close')
        client.request('PUT','/pools/'+pid,body)
        if not subset(client.request('GET','/pools/'+pid),body):raise RuntimeError('Quality update not verified: '+key)
        if was_open:transition(client,'pools',pid,'open')
        restrictions=client.all('user-restrictions',pool_id=pid)
        result.append({'stage':key,'exam_pass':c['review_exam_pass'],'control_pass':c['review_control_pass'],
                       'existing_restrictions':restrictions,'note':'Existing restrictions persist; inspect their reasons before removing any.'})
    return result
