import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import workflow as w


class Platform:
    def __init__(self):
        self.remote={k:{} for k in ('projects','skills','trainings','pools','tasks')};self.calls=[]
    def request(self,method,path,body=None,params=None):
        self.calls.append((method,path,copy.deepcopy(body)))
        if path=='/requester':return {'id':'account'}
        resource,*tail=path.strip('/').split('/')
        if method=='GET':return copy.deepcopy(self.remote[resource][tail[0]])
        if method=='POST' and resource=='tasks':
            for task in body:
                key='t'+str(len(self.remote['tasks']));self.remote['tasks'][key]={**copy.deepcopy(task),'id':key}
            return {}
        if method=='POST':
            key=resource+str(len(self.remote[resource]));r={**copy.deepcopy(body),'id':key,'status':'CLOSED'};self.remote[resource][key]=r;return copy.deepcopy(r)
        if method=='PUT':self.remote[resource][tail[0]].update(body);return copy.deepcopy(self.remote[resource][tail[0]])
        if method=='PATCH':self.remote[resource][tail[0]].update(body);return copy.deepcopy(self.remote[resource][tail[0]])
        raise AssertionError((method,path))
    def all(self,resource,**params):
        if resource=='assignments':return []
        return [copy.deepcopy(v) for v in self.remote[resource].values() if all(v.get(k)==a for k,a in params.items())]


class WorkflowTests(unittest.TestCase):
    def test_prepare_is_closed_complete_and_idempotent(self):
        real=w.BASE
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            import shutil
            shutil.copytree(real/'assets',root/'assets');c=w.config()
            c['instructions']={role:str((real/c['instructions'][role]).resolve()) for role in ('bbox','review')}
            w.save(root/'config.json',c)
            with patch.object(w,'BASE',root):
                api=Platform();w.prepare(api);first=copy.deepcopy(api.remote);w.prepare(api)
                self.assertEqual(api.remote,first)
                self.assertEqual(len(api.remote['projects']),2)
                self.assertEqual(len(api.remote['pools']),8)
                self.assertTrue(all(p['status']=='CLOSED' for p in api.remote['pools'].values()))
                self.assertFalse(any('/open' in p for _,p,_ in api.calls))
                state,_=w.state_for(api,'new');ids=w.ids_for(state)
                for stage in ('bbox_exam','review_exam'):
                    tasks=api.all('tasks',pool_id=ids[stage]);self.assertEqual(len(tasks),10)
                    self.assertTrue(all(t['overlap']==20 and not t['infinite_overlap'] for t in tasks))
                self.assertEqual(len(api.all('tasks',pool_id=ids['bbox_main'])),375)
                self.assertEqual(len(api.all('tasks',pool_id=ids['bbox_pilot'])),25)
                self.assertEqual(len(api.all('tasks',pool_id=ids['review_main'])),27)
                before=copy.deepcopy(api.remote['pools'][ids['review_main']]);w.update_settings(api,'new','review_main',price=.12)
                self.assertEqual(api.remote['pools'][ids['review_main']],{**before,'reward_per_assignment':.12})
                w.update_settings(api,'new','review_exam',overlap=40)
                self.assertTrue(all(t['overlap']==40 for t in api.all('tasks',pool_id=ids['review_exam'])))
    def test_read_only_and_invalid_geometry(self):
        with self.assertRaises(PermissionError):w.Client('fake').request('POST','/pools',{})
        for output in [{'status':'HAS_BUTTERFLY','result':[]},{'status':'NO_BUTTERFLY','result':[{}]},{'status':'HAS_BUTTERFLY','result':[{'shape':'rectangle','left':.9,'top':0,'width':.5,'height':.2}]}]:
            with self.assertRaises((ValueError,KeyError)):w.output_boxes(output)
    def test_dedup_votes_controls_and_self_review(self):
        task={'id':'t','input_values':{'mode':'bbox','photo_id':'1','item_key':'x','image':'url'}}
        source={'id':'a','user_id':'author','status':'SUBMITTED','submitted':'2026','tasks':[task],'solutions':[{'output_values':{'status':'NO_BUTTERFLY','result':[]}}]}
        def vote(worker,aid):
            iv={'annotation_id':'bbox_a_t_v1','source_assignment_id':'a','source_task_id':'t','source_status':'NO_BUTTERFLY','result':[]}
            return {'id':aid,'user_id':worker,'status':'SUBMITTED','tasks':[{'id':aid,'input_values':iv}],'solutions':[{'output_values':{'verdict':'BAD'}}]}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);w.save(p/'bbox_main_raw.json',{'assignments':[source]});w.save(p/'review_main_raw.json',{'assignments':[vote('author','s'),vote('r1','one'),vote('r1','duplicate'),vote('r2','two')]})
            rows=w.review_decisions(p);self.assertEqual(rows[0]['votes'],{'BAD':2});self.assertEqual(rows[0]['decision'],'REWORK')

if __name__=='__main__':unittest.main()
