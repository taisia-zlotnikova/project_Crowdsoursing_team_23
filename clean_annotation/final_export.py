"""Reproducible final export from frozen reviews and explicit requester corrections."""
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from types import SimpleNamespace
from workflow import output_boxes, save, digest

def normalize_output(output):
    boxes=output_boxes(output)
    return {'status':output['status'],'boxes':[{k:b[k] for k in ('left','top','width','height')} for b in boxes], 'requires_adjudication':False}

def jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))

ya=SimpleNamespace(normalize_output=normalize_output,save_json=save,save_jsonl=jsonl,fingerprint=digest)


def export(root):
    root = Path(root)
    base = root/'deliverables/annotation'
    report = json.loads((base/'decisions/latest.json').read_text())
    sources={}
    for file in (base/'raw').glob('*.json'):
        for a in json.loads(file.read_text())['assignments']:
            for t,solution in zip(a.get('tasks',[]),a.get('solutions',[])):
                if t['input_values'].get('mode') != 'bbox':continue
                aid=f"bbox_{a['id']}_{t['id']}_v1"
                sources[aid]={'annotation_id':aid,'pool':str(a['pool_id']),'worker_id':a['user_id'],
                    'image':t['input_values']['image'],'output_values':solution['output_values']}
    manifest={str(r['photo_id']):r for r in map(json.loads,(base/'dataset_manifest.jsonl').read_text().splitlines())}
    audit={r['annotation_id']:r for r in json.loads((base/'decisions/manual_decisions.json').read_text())['answers']}
    mapping={str(r['id']):r for r in json.loads((base/'final_corrections_mapping.json').read_text())}
    corrections = {}
    for r in json.loads((base/'final_corrections.json').read_text()):
        key = str(r['task_id'])
        if key not in mapping: raise ValueError('Unknown correction task ID')
        photo = mapping[key]['photo_id']
        if photo in corrections: raise ValueError('Duplicate correction')
        ya.normalize_output(r['output_values'])
        corrections[photo] = r['output_values']
    if set(corrections) != {'180452596','297914426','745859461'}: raise ValueError('Need exactly three requested corrections')
    out = root/'exports/final_butterflies_v1'
    (out/'yolo/labels').mkdir(parents=True,exist_ok=True)
    records, evidence, coco_images, coco_boxes = [], [], [], []
    for decision in sorted(report['photos'], key=lambda r:int(r['photo_id'])):
        photo = str(decision['photo_id']); source = sources[decision['annotation_id']]
        metadata = manifest[photo]
        if source['image'] != metadata['image_url']: raise ValueError('Source URL differs from manifest: '+photo)
        output = source['output_values']; origin = 'review_majority'
        override = audit.get(source['annotation_id'])
        if override:
            if override['output_sha256'] != ya.fingerprint(output): raise ValueError('Manual source hash differs')
            if override.get('corrected_output_values'):
                output = override['corrected_output_values']; origin = 'requester_corrected_status'
            elif override['verdict'] == 'GOOD': origin = 'requester_approved'
        if photo in corrections:
            if mapping[next(k for k,v in mapping.items() if v['photo_id']==photo)]['image'] != source['image']: raise ValueError('Correction image differs')
            output = corrections[photo]; origin = 'requester_redrawn'
        elif origin == 'review_majority' and decision['status'] != 'OK':
            raise ValueError('Unresolved annotation: '+photo)
        normalized = ya.normalize_output(output)
        if normalized['requires_adjudication']: raise ValueError('Unresolved technical status')
        dims = metadata['actual_dimensions']; width,height = dims['width'],dims['height']
        if width<=0 or height<=0: raise ValueError('Missing image dimensions')
        boxes = normalized['boxes']; pixel = [[b['left']*width,b['top']*height,b['width']*width,b['height']*height] for b in boxes]
        record = {'photo_id':photo,'image_url':source['image'],'file_name':photo+'.jpg','width':width,'height':height,
                  'status':normalized['status'],'category':'butterfly','boxes_normalized_xywh':[[b[k] for k in ('left','top','width','height')] for b in boxes],
                  'boxes_pixel_xywh':pixel,'decision_source':origin,'photo_license':metadata['photo_license'],
                  'photo_attribution':metadata.get('photo_attribution'),'observation_url':metadata.get('observation_url')}
        records.append(record)
        evidence.append({'photo_id':photo,'source_annotation_id':source['annotation_id'],'source_pool':source['pool'],
                         'source_worker_id':source['worker_id'],'source_output_sha256':ya.fingerprint(source['output_values']),
                         'final_output_sha256':ya.fingerprint(output),'decision_source':origin,'review_decision':decision,'manual_decision':override})
        image_id = int(photo)
        coco_images.append({'id':image_id,'file_name':record['file_name'],'width':width,'height':height,'coco_url':source['image']})
        for box in pixel:
            coco_boxes.append({'id':len(coco_boxes)+1,'image_id':image_id,'category_id':1,'bbox':box,'area':box[2]*box[3],'iscrowd':0})
        lines = ['0 '+' '.join(f'{v:.12f}' for v in (b[0]+b[2]/2,b[1]+b[3]/2,b[2],b[3])) for b in record['boxes_normalized_xywh']]
        (out/'yolo/labels'/f'{photo}.txt').write_text('\n'.join(lines)+ ('\n' if lines else ''))
    if len(records)!=400 or {r['photo_id'] for r in records}!=set(manifest):raise ValueError('Dataset does not match all 400 manifest photos')
    ya.save_jsonl(out/'annotations.jsonl',records)
    ya.save_json(out/'annotations.json',records)
    with (out/'annotations.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader()
        for r in records:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in r.items()})
    ya.save_json(out/'coco.json',{'info':{'description':'Butterfly bbox dataset, final v1; image dimensions from validated collection manifest'},'images':coco_images,'annotations':coco_boxes,'categories':[{'id':1,'name':'butterfly'}]})
    (out/'yolo/README.md').write_text('Class 0 = butterfly. Labels: normalized center_x center_y width height. Empty labels are negative images.\nImages are not bundled. Download from ../images_manifest.csv and save as <photo_id>.jpg in images/. No train/val split has been assigned.\n')
    with (out/'images_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['photo_id','file_name','image_url','width','height','photo_license']);writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in records)
    stats={'images':400,'boxes':len(coco_boxes),'statuses':dict(Counter(r['status'] for r in records)),
           'decision_sources':dict(Counter(r['decision_source'] for r in records)),
           'review_snapshot_utc':report['downloaded_at_utc'],'coordinate_convention':'xywh; normalized relative to width/height; pixels in COCO',
           'image_files_included':False}
    ya.save_json(out/'validation_report.json',stats)
    ya.save_json(base/'final_export/provenance.json',evidence)
    # A single HTML file; images load from their public source URLs.
    data=json.dumps(records,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Бабочки — финальная разметка 400 фото</title><style>
body{font:16px system-ui;margin:24px;background:#f4f6f8}header{position:sticky;top:0;background:#fff;padding:14px;z-index:2;border-radius:10px}input,select{font:inherit;padding:8px;margin:4px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px;margin-top:18px}article{background:white;padding:14px;border-radius:10px}.picture{position:relative}.picture img{width:100%;display:block}.picture svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}small{color:#526070}details pre{white-space:pre-wrap;font-size:12px}</style>
<header><b>Финальная разметка · 400 фото</b><br><input id="search" placeholder="ID фотографии"><select id="status"><option value="">Все ответы</option><option>HAS_BUTTERFLY</option><option>NO_BUTTERFLY</option></select><select id="origin"><option value="">Все источники решения</option><option>review_majority</option><option>requester_approved</option><option>requester_corrected_status</option><option>requester_redrawn</option></select><span id="count"></span><br><small>Рамки — финальные. Изображения загружаются из интернета. Нажмите фото, чтобы открыть оригинал.</small></header><main></main><script>
const data=DATA;const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){const rows=data.filter(r=>r.photo_id.includes(document.querySelector('#search').value.trim())&&(!statusEl.value||r.status===statusEl.value)&&(!originEl.value||r.decision_source===originEl.value));document.querySelector('#count').textContent='Показано: '+rows.length;document.querySelector('main').innerHTML=rows.map(r=>`<article><h3>Фото ${esc(r.photo_id)} · ${r.status==='HAS_BUTTERFLY'?'Бабочка есть':'Бабочки нет'}</h3><small>${esc(r.decision_source)} · рамок: ${r.boxes_normalized_xywh.length}</small><div class="picture"><a href="${esc(r.image_url)}" target="_blank" rel="noopener"><img loading="lazy" src="${esc(r.image_url)}" width="${r.width}" height="${r.height}" style="height:auto" alt="Фото ${esc(r.photo_id)}"></a><svg viewBox="0 0 1000 1000" preserveAspectRatio="none">${r.boxes_normalized_xywh.map(b=>`<rect x="${b[0]*1000}" y="${b[1]*1000}" width="${b[2]*1000}" height="${b[3]*1000}" fill="none" stroke="#00e5ff" stroke-width="3" vector-effect="non-scaling-stroke"/>`).join('')}</svg></div><details><summary>Координаты и метаданные</summary><pre>${esc(JSON.stringify(r,null,2))}</pre></details></article>`).join('')}
const statusEl=document.querySelector('#status'),originEl=document.querySelector('#origin');document.querySelector('#search').oninput=render;statusEl.onchange=render;originEl.onchange=render;render();</script></html>'''.replace('DATA',data)
    (out/'gallery.html').write_text(page)
    (out/'README.md').write_text('''# Финальный датасет бабочек v1
400 уникальных фото, один класс butterfly. Это объединение пилота (25) и основного набора (375), не экзаменов и не ханипотов.

- annotations.jsonl / annotations.json: запись на изображение; нормализованные и пиксельные XYWH.
- annotations.csv: одна строка на изображение, списки рамок записаны JSON.
- coco.json: стандартные COCO detection images/annotations/categories, category_id=1; координаты в пикселях.
- yolo/labels/*.txt: class_id=0, нормализованные center_x center_y width height; отрицательные фото имеют пустой txt.
- images_manifest.csv: ссылки и соответствие имён локальных изображений.
- gallery.html: просмотр финальных рамок, поиск ID и фильтры.
- validation_report.json: контроль полноты и статистика.

Изображения не включены в архив; ссылки публичные, для HTML нужен интернет. Размеры взяты из actual_dimensions проверенного исходного манифеста; при экспорте изображения повторно не скачивались. Для COCO/YOLO скачайте изображения по манифесту с указанными именами. Разбиение train/val/test здесь не назначалось.

Выбор версии: последняя размеченная версия по сохранённой итоговой проверке, ручные решения заказчика имеют приоритет. Три последние рамки взяты из final_corrections.json; для 743433072 вручную задан NO_BUTTERFLY. Нерешённые технические ответы и ошибки координат в финал не допускаются. Проверки подтверждают полноту, согласованность и корректность координат; это не независимая экспертная переоценка всех изображений.
Подробное происхождение с идентификаторами исполнителей хранится отдельно в platform/private/.../final_export/provenance.json и в публичный архив не включено.
''')
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    ya.save_json(out/'SHA256SUMS.json',hashes)
    archive=out.parent/(out.name+'.zip')
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(out.parent)))
    print(json.dumps(stats,ensure_ascii=False,indent=2));print(archive)
    return out

if __name__=='__main__':
    export(Path(__file__).resolve().parents[1])
